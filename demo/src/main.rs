use std::env;
use std::fs::{self, File};
use std::future::Future;
use std::io::BufWriter;
use std::path::PathBuf;
use std::process::exit;
use std::sync::{Arc, Mutex};
use std::thread::{self, sleep};
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
use capture::{AudioPacket, SendAudioPacket};

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

    let cli_model = args.model.clone();
    if let Some(p) = cli_model.as_ref() {
        log::info!("Model specified via --model: {}", p.display());
    } else if let Ok(env_p) = env::var("DF_MODEL") {
        log::info!("Model specified via DF_MODEL: {}", PathBuf::from(env_p).display());
    } else {
        #[cfg(feature = "default-model")]
        let compiled_default: &str = "DeepFilterNet3_onnx.tar.gz";
    }

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

struct WavWriterThread {
    writer: WavWriter<BufWriter<File>>,
}

impl WavWriterThread {
    fn new(path: &str, spec: &hound::WavSpec) -> Result<Self, String> {
        let file = File::create(path).map_err(|e| format!("Failed to create {}: {:?}", path, e))?;
        let buf_writer = BufWriter::with_capacity(32768, file); // 큰 버퍼 사용
        let writer = WavWriter::new(buf_writer, *spec)
            .map_err(|e| format!("Failed to create WAV writer: {:?}", e))?;
        log::info!("Created WAV writer: {}", path);
        Ok(Self { writer })
    }

    fn write_samples(&mut self, samples: &[f32]) -> Result<(), String> {
        for &sample in samples {
            self.writer.write_sample(sample)
                .map_err(|e| format!("Write error: {:?}", e))?;
        }
        Ok(())
    }

    fn finalize(self) -> Result<(), String> {
        self.writer.finalize()
            .map_err(|e| format!("Finalize error: {:?}", e))
    }
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
    r_audio: Option<crossbeam_channel::Receiver<AudioPacket>>,
    s_controls: Option<SendControl>,
    recording_state: RecordingState,
    writer_thread_handle: Option<thread::JoinHandle<()>>,
    writer_thread_sender: Option<crossbeam_channel::Sender<WriterCommand>>,
    sample_rate: usize,
    current_basename: Option<String>,
}

enum WriterCommand {
    WriteRaw(Vec<f32>),
    WriteEnh(Vec<f32>),
    Stop,
}

#[derive(Debug, Clone)]
pub enum Message {
    None,
    Tick,
    LsnrChanged(f32),
    NoisyChanged,
    EnhChanged,
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
    fn w(&self) -> usize { self.n_frames as usize }
    fn h(&self) -> usize { self.n_freqs as usize }
    fn update<I>(&mut self, specs: I, mut n_specs: usize)
    where
        I: Iterator<Item = Box<[f32]>>,
    {
        if n_specs == 0 { return; }
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
                writer_thread_handle: None,
                writer_thread_sender: None,
                sample_rate: 48000,
                current_basename: None,
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
                // Writer 스레드 정리
                if let Some(sender) = self.writer_thread_sender.take() {
                    let _ = sender.send(WriterCommand::Stop);
                }
                if let Some(handle) = self.writer_thread_handle.take() {
                    let _ = handle.join();
                }

                if let Some(worker) = &mut self.df_worker {
                    worker.should_stop().expect("Failed to stop DF worker");
                }
                exit(0);
            }

            Message::StartRecording => {
                if self.recording_state == RecordingState::Stopped {
                    let (s_lsnr,  r_lsnr)  = unbounded();
                    let (s_noisy, r_noisy) = unbounded();
                    let (s_enh,   r_enh)   = unbounded();
                    let (s_audio, r_audio) = unbounded::<AudioPacket>();
                    let (s_controls, r_controls) = unbounded();

                    let env_model = env::var("DF_MODEL").ok().map(PathBuf::from);
                    let model_path = env_model.or_else(|| unsafe { MODEL_PATH.clone() });

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

                    fs::create_dir_all("record").expect("Failed to create record directory");
                    let ts = chrono::Local::now().format("%Y_%m_%d_%H%M%S").to_string();
                    let base = format!("record_{}", ts);
                    self.current_basename = Some(base.clone());

                    let spec = hound::WavSpec {
                        channels: 1,
                        sample_rate: self.sample_rate as u32,
                        bits_per_sample: 32,
                        sample_format: hound::SampleFormat::Float,
                    };

                    let raw_path = format!("record/record_raw_{}.wav", ts);
                    let enh_path = format!("record/record_enh_{}.wav", ts);

                    let (writer_tx, writer_rx) = unbounded::<WriterCommand>();

                    let handle = thread::spawn(move || {
                        let mut raw_writer = match WavWriterThread::new(&raw_path, &spec) {
                            Ok(w) => w,
                            Err(e) => {
                                log::error!("{}", e);
                                return;
                            }
                        };
                        let mut enh_writer = match WavWriterThread::new(&enh_path, &spec) {
                            Ok(w) => w,
                            Err(e) => {
                                log::error!("{}", e);
                                return;
                            }
                        };

                        log::info!("WAV writer thread started");

                        while let Ok(cmd) = writer_rx.recv() {
                            match cmd {
                                WriterCommand::WriteRaw(samples) => {
                                    if let Err(e) = raw_writer.write_samples(&samples) {
                                        log::error!("Failed to write RAW: {}", e);
                                    }
                                }
                                WriterCommand::WriteEnh(samples) => {
                                    if let Err(e) = enh_writer.write_samples(&samples) {
                                        log::error!("Failed to write ENH: {}", e);
                                    }
                                }
                                WriterCommand::Stop => break,
                            }
                        }

                        // Finalize all files
                        if let Err(e) = raw_writer.finalize() {
                            log::error!("Failed to finalize RAW: {}", e);
                        } else {
                            log::info!("Finalized RAW");
                        }
                        if let Err(e) = enh_writer.finalize() {
                            log::error!("Failed to finalize ENH: {}", e);
                        } else {
                            log::info!("Finalized ENH");
                        }

                        log::info!("WAV writer thread stopped");
                    });

                    self.writer_thread_handle = Some(handle);
                    self.writer_thread_sender = Some(writer_tx);

                    log::info!(
                        "Started recording: raw/enh/noise with basename {} (sr = {}, 32-bit float)",
                        base, self.sample_rate
                    );

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
                    // Writer 스레드 정리
                    if let Some(sender) = self.writer_thread_sender.take() {
                        let _ = sender.send(WriterCommand::Stop);
                    }
                    if let Some(handle) = self.writer_thread_handle.take() {
                        let _ = handle.join();
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
                    self.current_basename = None;

                    self.recording_state = RecordingState::Stopped;
                    log::info!("Stopped recording");
                }
            }

            Message::Tick => {
                if self.recording_state != RecordingState::Recording {
                    return Command::none();
                }
                let mut commands = Vec::new();
                if let Some(task) = self.update_lsnr() { commands.push(Command::perform(task, move |m| m)) }
                if let Some(task) = self.update_noisy() { commands.push(Command::perform(task, move |m| m)) }
                if let Some(task) = self.update_enh()   { commands.push(Command::perform(task, move |m| m)) }
                if let Some(task) = self.update_audio() { commands.push(Command::perform(task, move |m| m)) }
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


            Message::AttenLimChanged(v) => {
                self.atten_lim = v;
                if let Some(s) = &self.s_controls {
                    s.send((DfControl::AttenLim, v)).expect("Failed to send DfControl")
                }
            }
            Message::PostFilterChanged(v) => {
                self.post_filter_beta = v;
                if let Some(s) = &self.s_controls {
                    s.send((DfControl::PostFilterBeta, v)).expect("Failed to send DfControl")
                }
            }
            Message::MinThreshDbChanged(v) => {
                self.min_threshdb = v;
                if let Some(s) = &self.s_controls {
                    s.send((DfControl::MinThreshDb, v)).expect("Failed to send DfControl")
                }
            }
            Message::MaxErbThreshDbChanged(v) => {
                self.max_erbthreshdb = v;
                if let Some(s) = &self.s_controls {
                    s.send((DfControl::MaxErbThreshDb, v)).expect("Failed to send DfControl")
                }
            }
            Message::MaxDfThreshDbChanged(v) => {
                self.max_dfthreshdb = v;
                if let Some(s) = &self.s_controls {
                    s.send((DfControl::MaxDfThreshDb, v)).expect("Failed to send DfControl")
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
            if recv.is_empty() { return None; }
            let recv = recv.clone();
            Some(async move {
                sleep(Duration::from_millis(100));
                let mut lsnr = 0.;
                let mut n = 0;
                while let Ok(v) = recv.try_recv() {
                    lsnr += v; n += 1;
                }
                if n > 0 { Message::LsnrChanged(lsnr / n as f32) } else { Message::None }
            })
        } else { None }
    }

    fn update_noisy(&mut self) -> Option<impl Future<Output = Message>> {
        if let Some(recv) = &self.r_noisy {
            if recv.is_empty() { return None; }
            let recv = recv.clone();
            Some(async move {
                let n = recv.len();
                unsafe {
                    let mut spec = SPEC_NOISY.as_mut().unwrap().lock().expect("Failed to lock SPEC_NOISY");
                    spec.update(recv.iter().take(n), n);
                }
                Message::NoisyChanged
            })
        } else { None }
    }

    fn update_enh(&mut self) -> Option<impl Future<Output = Message>> {
        if let Some(recv) = &self.r_enh {
            if recv.is_empty() { return None; }
            let recv = recv.clone();
            Some(async move {
                let n = recv.len();
                unsafe {
                    let mut spec = SPEC_ENH.as_mut().unwrap().lock().expect("Failed to lock SPEC_ENH");
                    spec.update(recv.iter().take(n), n);
                }
                Message::EnhChanged
            })
        } else { None }
    }

    fn update_audio(&mut self) -> Option<impl Future<Output = Message>> {
        if let Some(recv) = &self.r_audio {
            if recv.is_empty() { return None; }
            let recv = recv.clone();
            let sender = self.writer_thread_sender.clone();
            
            Some(async move {
                // ==== FIXED: 모든 패킷을 Writer 스레드로 직접 전송 ====
                let mut packet_count = 0;
                while let Ok(pkt) = recv.try_recv() {
                    if let Some(ref s) = sender {
                        let min_len = pkt.raw.len().min(pkt.enh.len());
                        if min_len > 0 {
                            let raw = pkt.raw[..min_len].to_vec();
                            let enh = pkt.enh[..min_len].to_vec();
                            
                            // Writer 스레드로 전송 (논블로킹)
                            let _ = s.try_send(WriterCommand::WriteRaw(raw));
                            let _ = s.try_send(WriterCommand::WriteEnh(enh));
                            
                            packet_count += 1;
                        }
                    }
                }
                
                Message::None // 처리만 하고 UI 업데이트 없음
            })
        } else { None }
    }

    fn specs(&self) -> Container<Message> {
        container(column![
            spec_view("Noisy", self.noisy_img.clone(), 1000, 250),
            spec_view("Enhanced", self.enh_img.clone(), 1000, 250),
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