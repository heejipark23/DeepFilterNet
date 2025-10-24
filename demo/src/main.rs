use std::env;
use std::fs::{self, File};
use std::future::Future;
use std::io::BufWriter;
use std::path::PathBuf;
use std::process::exit;
use std::sync::{Arc, Mutex};
use std::thread::sleep;
use std::time::Duration;

use clap::{Parser, ValueHint};
use crossbeam_channel::unbounded;
use hound::WavWriter;
use iced::widget::{self, column, container, image, row, slider, text, Container, Image};
use iced::{
    alignment, executor, Alignment, Application, Command, ContentFit, Element, Length, Settings,
    Subscription, Theme,
};
use image_rs::{imageops, Rgba, RgbaImage};

mod capture;
mod cmap;
use capture::*;

/// Simple program to sample from a hd5 dataset directory
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to model tar.gz
    #[arg(short, long, value_hint = ValueHint::FilePath)]
    model: Option<PathBuf>,
    /// Logging verbosity
    #[arg(
        long,
        short = 'v',
        action = clap::ArgAction::Count,
        global = true,
        help = "Increase logging verbosity with multiple `-vv`",
    )]
    verbose: u8,
}

pub fn main() -> iced::Result {
    let args = Args::parse();
    let level = match args.verbose {
        0 => log::LevelFilter::Warn,
        1 => log::LevelFilter::Info,
        2 => log::LevelFilter::Debug,
        _ => log::LevelFilter::Trace,
    };
    let tract_level = match args.verbose {
        0..=3 => log::LevelFilter::Error,
        4 => log::LevelFilter::Info,
        5 => log::LevelFilter::Debug,
        _ => log::LevelFilter::Trace,
    };
    if args.model.is_some() {
        unsafe { MODEL_PATH = args.model }
    }

    capture::INIT_LOGGER.call_once(|| {
        env_logger::Builder::from_env(env_logger::Env::default())
            .filter_level(level)
            .filter_module("tract_onnx", tract_level)
            .filter_module("tract_hir", tract_level)
            .filter_module("tract_core", tract_level)
            .filter_module("tract_linalg", tract_level)
            .filter_module("iced_winit", log::LevelFilter::Error)
            .filter_module("iced_wgpu", log::LevelFilter::Error)
            .filter_module("wgpu_core", log::LevelFilter::Error)
            .filter_module("wgpu_hal", log::LevelFilter::Error)
            .filter_module("naga", log::LevelFilter::Error)
            .filter_module("crossfont", log::LevelFilter::Error)
            .filter_module("cosmic_text", log::LevelFilter::Error)
            .format(capture::log_format)
            .init();
    });

    SpecView::run(Settings::default())
}

static mut SPEC_NOISY: Option<Arc<Mutex<SpecImage>>> = None;
static mut SPEC_ENH: Option<Arc<Mutex<SpecImage>>> = None;

#[derive(Debug, Clone, Copy, PartialEq)]
enum RecordingState {
    Stopped,
    Recording,
}

struct SpecView {
    df_worker: Option<DeepFilterCapture>,
    lsnr: f32,
    atten_lim: f32,
    post_filter_beta: f32,
    min_threshdb: f32,
    max_erbthreshdb: f32,
    max_dfthreshdb: f32,
    noisy_img: image::Handle,
    enh_img: image::Handle,
    r_lsnr: Option<RecvLsnr>,
    r_noisy: Option<RecvSpec>,
    r_enh: Option<RecvSpec>,
    r_audio: Option<RecvAudio>,
    s_controls: Option<SendControl>,
    recording_state: RecordingState,
    wav_writer: Option<Arc<Mutex<Option<WavWriter<BufWriter<File>>>>>>,
    sample_rate: usize,
    current_filename: Option<String>,
}

#[derive(Debug, Clone)]
pub enum Message {
    None,
    Tick,
    LsnrChanged(f32),
    NoisyChanged,
    EnhChanged,
    AudioReceived(Vec<f32>),
    AttenLimChanged(f32),
    PostFilterChanged(f32),
    MinThreshDbChanged(f32),
    MaxErbThreshDbChanged(f32),
    MaxDfThreshDbChanged(f32),
    StartRecording,
    PauseRecording,
    Exit,
}

struct SpecImage {
    im: RgbaImage,
    n_frames: u32,
    n_freqs: u32,
    vmin: f32,
    vmax: f32,
}

impl SpecImage {
    fn new(n_frames: u32, n_freqs: u32, vmin: f32, vmax: f32) -> Self {
        Self {
            im: RgbaImage::new(n_freqs, n_frames),
            n_frames,
            n_freqs,
            vmin,
            vmax,
        }
    }
    fn w(&self) -> usize {
        self.n_frames as usize
    }
    fn h(&self) -> usize {
        self.n_freqs as usize
    }
    fn update<I>(&mut self, specs: I, mut n_specs: usize)
    where
        I: Iterator<Item = Box<[f32]>>,
    {
        if n_specs == 0 {
            return;
        }
        if n_specs >= self.n_frames as usize {
            n_specs = self.n_frames as usize - 1;
        }
        for (spec, im_row) in specs.take(n_specs).zip(self.im.rows_mut()) {
            for (s, x) in spec.iter().zip(im_row) {
                let v = (s.min(self.vmax).max(self.vmin) - self.vmin) / (self.vmax - self.vmin);
                *x = Rgba(cmap::CMAP_INFERNO[(v * 255.) as usize]);
            }
        }
        let (w, h) = (self.w(), self.h());
        self.im.rotate_left((w - n_specs) * 4 * h);
    }
    fn image_handle(&self) -> image::Handle {
        let imt_buf = imageops::rotate270(&self.im).as_raw().to_vec();
        image::Handle::from_pixels(self.n_frames, self.n_freqs, imt_buf)
    }
}

impl Application for SpecView {
    type Executor = executor::Default;
    type Message = Message;
    type Theme = Theme;
    type Flags = ();

    fn new(_flags: ()) -> (Self, Command<Message>) {
        let w = 1000;
        let h = 250;
        let (noisy_img, enh_img) = unsafe {
            SPEC_NOISY = Some(Arc::new(Mutex::new(SpecImage::new(w, h, -100., -10.))));
            SPEC_ENH = Some(Arc::new(Mutex::new(SpecImage::new(w, h, -100., -10.))));
            (
                SPEC_NOISY.as_ref().unwrap().lock().unwrap().image_handle(),
                SPEC_ENH.as_ref().unwrap().lock().unwrap().image_handle(),
            )
        };
        
        (
            Self {
                df_worker: None,
                lsnr: 0.,
                atten_lim: 100.,
                post_filter_beta: 0.,
                min_threshdb: -15.,
                max_erbthreshdb: 35.,
                max_dfthreshdb: 35.,
                r_lsnr: None,
                r_noisy: None,
                r_enh: None,
                r_audio: None,
                s_controls: None,
                noisy_img,
                enh_img,
                recording_state: RecordingState::Stopped,
                wav_writer: None,
                sample_rate: 48000,
                current_filename: None,
            },
            Command::none(),
        )
    }

    fn title(&self) -> String {
        "Robot Ego-Noise Denoising Demo".to_string()
    }

    fn update(&mut self, message: Message) -> Command<Message> {
        match message {
            Message::None => (),
            Message::Exit => {
                // Finalize WAV file if recording
                if let Some(writer) = self.wav_writer.take() {
                    if let Ok(mut guard) = writer.lock() {
                        if let Some(w) = guard.take() {
                            if let Err(e) = w.finalize() {
                                log::error!("Failed to finalize WAV file: {:?}", e);
                            }
                        }
                    }
                }

                if let Some(worker) = &mut self.df_worker {
                    worker.should_stop().expect("Failed to stop DF worker");
                }
                exit(0);
            }
            Message::StartRecording => {
                if self.recording_state == RecordingState::Stopped {
                    let (s_lsnr, r_lsnr) = unbounded();
                    let (s_noisy, r_noisy) = unbounded();
                    let (s_enh, r_enh) = unbounded();
                    let (s_audio, r_audio) = unbounded();
                    let (s_controls, r_controls) = unbounded();

                    let model_path = env::var("DF_MODEL").ok().map(PathBuf::from);
                    let df_worker = DeepFilterCapture::new(
                        model_path,
                        Some(s_lsnr),
                        Some(s_noisy),
                        Some(s_enh),
                        Some(s_audio),
                        Some(r_controls),
                    )
                    .expect("Failed to initialize DeepFilterNet audio capturing");

                    self.sample_rate = df_worker.sr;
                    
                    // Create WAV writer
                    let timestamp = chrono::Local::now().format("%Y_%m_%d_%H%M%S");
                    let filename = format!("Record_{}.wav", timestamp);
                    fs::create_dir_all("record").expect("Failed to create record directory");
                    let filepath = format!("record/{}", filename);
                    
                    let spec = hound::WavSpec {
                        channels: 1,
                        sample_rate: self.sample_rate as u32,
                        bits_per_sample: 16,
                        sample_format: hound::SampleFormat::Int,
                    };
                    
                    if let Ok(file) = File::create(&filepath) {
                        let writer = WavWriter::new(BufWriter::new(file), spec)
                            .expect("Failed to create WAV writer");
                        self.wav_writer = Some(Arc::new(Mutex::new(Some(writer))));
                        self.current_filename = Some(filename.clone());
                        log::info!("Started recording to {}", filepath);
                    }
                    
                    self.df_worker = Some(df_worker);
                    self.r_lsnr = Some(r_lsnr);
                    self.r_noisy = Some(r_noisy);
                    self.r_enh = Some(r_enh);
                    self.r_audio = Some(r_audio);
                    self.s_controls = Some(s_controls);
                    
                    self.recording_state = RecordingState::Recording;
                }
            }
            Message::PauseRecording => {
                if self.recording_state == RecordingState::Recording {
                    // Finalize and close WAV file
                    // Finalize and close WAV file
                    if let Some(arc) = self.wav_writer.take() {
                        match arc.lock() {
                            Ok(mut guard) => {
                                if let Some(wav) = guard.take() {
                                    match wav.finalize() {
                                        Ok(_) => {
                                            if let Some(filename) = &self.current_filename {
                                                log::info!("Saved recording to record/{}", filename);
                                            }
                                        }
                                        Err(e) => log::error!("Failed to finalize WAV file: {:?}", e),
                                    }
                                } else {
                                    log::warn!("PauseRecording: WAV writer already finalized");
                                }
                            }
                            Err(e) => {
                                log::error!("PauseRecording: failed to lock WAV writer: {:?}", e);
                            }
                        }
                    }
                    
                    if let Some(worker) = &mut self.df_worker {
                        worker.should_stop().expect("Failed to stop DF worker");
                    }
                    self.df_worker = None;
                    self.r_lsnr = None;
                    self.r_noisy = None;
                    self.r_enh = None;
                    self.r_audio = None;
                    self.s_controls = None;
                    self.current_filename = None;
                    
                    self.recording_state = RecordingState::Stopped;
                    log::info!("Stopped recording");
                }
            }
            Message::Tick => {
                if self.recording_state != RecordingState::Recording {
                    return Command::none();
                }
                
                let mut commands = Vec::new();
                if let Some(task) = self.update_lsnr() {
                    commands.push(Command::perform(task, move |message| message))
                }
                if let Some(task) = self.update_noisy() {
                    commands.push(Command::perform(task, move |message| message))
                }
                if let Some(task) = self.update_enh() {
                    commands.push(Command::perform(task, move |message| message))
                }
                if let Some(task) = self.update_audio() {
                    commands.push(Command::perform(task, move |message| message))
                }
                return Command::batch(commands);
            }
            Message::LsnrChanged(lsnr) => self.lsnr = lsnr,
            Message::NoisyChanged => {
                self.noisy_img = unsafe {
                    SPEC_NOISY
                        .as_ref()
                        .unwrap()
                        .lock()
                        .expect("Failed to lock SPEC_NOISY")
                        .image_handle()
                };
            }
            Message::EnhChanged => {
                self.enh_img = unsafe {
                    SPEC_ENH
                        .as_ref()
                        .unwrap()
                        .lock()
                        .expect("Failed to lock SPEC_ENH")
                        .image_handle()
                };
            }
            Message::AudioReceived(samples) => {
                // Write samples to WAV file in real-time
                if let Some(writer) = &self.wav_writer {
                    if let Ok(mut guard) = writer.lock() {
                        if let Some(w) = guard.as_mut() {
                            for &sample in &samples {
                                let sample_i16 = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
                                if let Err(e) = w.write_sample(sample_i16) {
                                    log::error!("Failed to write sample: {:?}", e);
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            Message::AttenLimChanged(v) => {
                self.atten_lim = v;
                if let Some(s) = &self.s_controls {
                    s.send((DfControl::AttenLim, v))
                        .expect("Failed to send DfControl")
                }
            }
            Message::PostFilterChanged(v) => {
                self.post_filter_beta = v;
                if let Some(s) = &self.s_controls {
                    s.send((DfControl::PostFilterBeta, v))
                        .expect("Failed to send DfControl")
                }
            }
            Message::MinThreshDbChanged(v) => {
                self.min_threshdb = v;
                if let Some(s) = &self.s_controls {
                    s.send((DfControl::MinThreshDb, v))
                        .expect("Failed to send DfControl")
                }
            }
            Message::MaxErbThreshDbChanged(v) => {
                self.max_erbthreshdb = v;
                if let Some(s) = &self.s_controls {
                    s.send((DfControl::MaxErbThreshDb, v))
                        .expect("Failed to send DfControl")
                }
            }
            Message::MaxDfThreshDbChanged(v) => {
                self.max_dfthreshdb = v;
                if let Some(s) = &self.s_controls {
                    s.send((DfControl::MaxDfThreshDb, v))
                        .expect("Failed to send DfControl")
                }
            }
        }
        Command::none()
    }

    fn view(&self) -> Element<Message> {
        let title_row = row![
            text("Robot Ego-Noise Denoising Demo").size(40).width(Length::Fill),
            button("exit").on_press(Message::Exit)
        ]
        .spacing(10)
        .width(1000);

        let content = column![title_row];

        #[cfg(feature = "thresholds")]
        let content = {
            content
                .push(slider_view(
                    "Threshold Min [dB]",
                    self.min_threshdb,
                    -15.,
                    35.,
                    Message::MinThreshDbChanged,
                    1000,
                    0,
                    3.,
                ))
                .push(slider_view(
                    "Threshold ERB Max [dB]",
                    self.max_erbthreshdb,
                    -15.,
                    35.,
                    Message::MaxErbThreshDbChanged,
                    1000,
                    0,
                    3.,
                ))
                .push(slider_view(
                    "Threshold DF  Max [dB]",
                    self.max_dfthreshdb,
                    -15.,
                    35.,
                    Message::MaxDfThreshDbChanged,
                    1000,
                    0,
                    3.,
                ))
        };
        let content = content
            .push(slider_view(
                "Noise Attenuation [dB]",
                self.atten_lim,
                0.,
                100.,
                Message::AttenLimChanged,
                1000,
                0,
                3.,
            ))
            .push(
                row![
                    button("Start").on_press(Message::StartRecording),
                    button("Pause").on_press(Message::PauseRecording),
                    if self.recording_state == RecordingState::Recording {
                        text("● Recording...")
                            .size(20)
                            .style(iced::theme::Text::Color(iced::Color::from_rgb(1.0, 0.0, 0.0)))
                    } else {
                        text("")
                    }
                ]
                .spacing(10)
                .align_items(Alignment::Center)
            )
            .push(self.specs())
            .push(
                row![
                    text("Current SNR:").size(18),
                    text(format!("{:>5.1} dB", self.lsnr))
                        .size(18)
                        .width(80)
                        .horizontal_alignment(alignment::Horizontal::Right)
                ]
                .spacing(20)
                .align_items(Alignment::End),
            );

        container(content)
            .padding(50)
            .width(Length::Fill)
            .height(Length::Fill)
            .center_x()
            .center_y()
            .into()
    }

    fn subscription(&self) -> Subscription<Message> {
        iced::time::every(std::time::Duration::from_millis(20)).map(|_| Message::Tick)
    }
}

impl SpecView {
    fn update_lsnr(&mut self) -> Option<impl Future<Output = Message>> {
        if let Some(recv) = &self.r_lsnr {
            if recv.is_empty() {
                return None;
            }
            let recv = recv.clone();
            Some(async move {
                sleep(Duration::from_millis(100));
                let mut lsnr = 0.;
                let mut n = 0;
                while let Ok(v) = recv.try_recv() {
                    lsnr += v;
                    n += 1;
                }
                if n > 0 {
                    lsnr /= n as f32;
                    Message::LsnrChanged(lsnr)
                } else {
                    Message::None
                }
            })
        } else {
            None
        }
    }

    fn update_noisy(&mut self) -> Option<impl Future<Output = Message>> {
        if let Some(recv) = &self.r_noisy {
            if recv.is_empty() {
                return None;
            }
            let recv = recv.clone();
            Some(async move {
                let n = recv.len();
                unsafe {
                    let mut spec =
                        SPEC_NOISY.as_mut().unwrap().lock().expect("Failed to lock SPEC_NOISY");
                    spec.update(recv.iter().take(n), n);
                }
                Message::NoisyChanged
            })
        } else {
            None
        }
    }

    fn update_enh(&mut self) -> Option<impl Future<Output = Message>> {
        if let Some(recv) = &self.r_enh {
            if recv.is_empty() {
                return None;
            }
            let recv = recv.clone();
            
            Some(async move {
                let n = recv.len();
                unsafe {
                    let mut spec = SPEC_ENH.as_mut().unwrap().lock().expect("Failed to lock SPEC_ENH");
                    spec.update(recv.iter().take(n), n);
                }
                Message::EnhChanged
            })
        } else {
            None
        }
    }

    fn update_audio(&mut self) -> Option<impl Future<Output = Message>> {
        if let Some(recv) = &self.r_audio {
            if recv.is_empty() {
                return None;
            }
            let recv = recv.clone();
            
            Some(async move {
                let mut all_samples = Vec::new();
                while let Ok(samples) = recv.try_recv() {
                    all_samples.extend_from_slice(&samples);
                }
                if !all_samples.is_empty() {
                    Message::AudioReceived(all_samples)
                } else {
                    Message::None
                }
            })
        } else {
            None
        }
    }

    fn specs(&self) -> Container<Message> {
        container(column![
            spec_view("Noisy", self.noisy_img.clone(), 1000, 250),
            spec_view("DeepFilterNet Enhanced", self.enh_img.clone(), 1000, 250),
        ])
    }
}

fn spec_view(title: &str, im: image::Handle, width: u16, height: u16) -> Element<Message> {
    column![
        text(title).size(24).width(Length::Fill),
        spec_raw(im, width, height)
    ]
    .max_width(width)
    .width(Length::Fill)
    .into()
}

fn spec_raw<'a>(im: image::Handle, width: u16, height: u16) -> Container<'a, Message> {
    container(Image::new(im).width(width).height(height).content_fit(ContentFit::Fill))
        .max_width(width)
        .max_height(height)
        .width(Length::Fill)
        .center_x()
        .center_y()
}

#[allow(clippy::too_many_arguments)]
fn slider_view<'a>(
    title: &str,
    value: f32,
    min: f32,
    max: f32,
    message: impl Fn(f32) -> Message + 'a,
    width: u16,
    precision: usize,
    step: f32,
) -> Element<'a, Message> {
    column![
        text(title).size(18).width(Length::Fill),
        row![
            container(slider(min..=max, value, message).step(step)).width(Length::Fill),
            text(format!("{:.precision$}", value))
                .size(18)
                .width(100)
                .horizontal_alignment(alignment::Horizontal::Right)
                .vertical_alignment(alignment::Vertical::Top),
        ]
    ]
    .max_width(width)
    .width(Length::Fill)
    .into()
}

fn button(text: &str) -> widget::Button<'_, Message> {
    widget::button(text).padding(10)
}