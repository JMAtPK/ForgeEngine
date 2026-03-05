//! Game-mode harness — windowing, input capture, render dispatch.

use crate::dag_runner::{self, DagRunner, GameManifest};
use crate::gpu;
use crate::registry;
use crate::schema::to_fix16;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use wgpu::Backends;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::EventLoop;
use winit::keyboard::{Key, NamedKey};
use winit::window::Window;
use winit::platform::x11::EventLoopBuilderExtX11;

// ===== Hardcoded buffer layouts =====

fn pack_input_buffer(keys_down: u32, mouse_x: i32, mouse_y: i32, mouse_buttons: u32) -> [u8; 16] {
    let mut buf = [0u8; 16];
    buf[0..4].copy_from_slice(&keys_down.to_le_bytes());
    buf[4..8].copy_from_slice(&mouse_x.to_le_bytes());
    buf[8..12].copy_from_slice(&mouse_y.to_le_bytes());
    buf[12..16].copy_from_slice(&mouse_buttons.to_le_bytes());
    buf
}

fn pack_globals_buffer(
    frame_number: u32,
    delta_time: i32,
    world_width: i32,
    world_height: i32,
    random_seed: u32,
) -> [u8; 32] {
    let mut buf = [0u8; 32];
    buf[0..4].copy_from_slice(&frame_number.to_le_bytes());
    buf[4..8].copy_from_slice(&delta_time.to_le_bytes());
    buf[8..12].copy_from_slice(&world_width.to_le_bytes());
    buf[12..16].copy_from_slice(&world_height.to_le_bytes());
    buf[16..20].copy_from_slice(&random_seed.to_le_bytes());
    buf
}

// ===== Harness State =====

struct HarnessState {
    // Input
    keys_down: u32,
    mouse_x: f64,
    mouse_y: f64,
    mouse_buttons: u32,

    // Timing
    last_frame: Instant,
    frame_number: u32,
    max_delta_time_fix16: i32,

    // World params (fix16)
    world_width_fix16: i32,
    world_height_fix16: i32,

    // PRNG
    rng: SmallRng,

    // Config
    manifest_path: PathBuf,
    registry_dir: PathBuf,
    backends: Backends,
    verify: bool,
    render_kernel_name: Option<String>,

    // Runtime (initialized on resumed)
    window: Option<Arc<Window>>,
    surface: Option<wgpu::Surface<'static>>,
    surface_width: u32,
    surface_height: u32,
    runner: Option<DagRunner>,
}

fn key_bit(key: &Key) -> Option<u32> {
    match key {
        Key::Character(c) => match c.as_str() {
            "w" | "W" => Some(0),
            "a" | "A" => Some(1),
            "s" | "S" => Some(2),
            "d" | "D" => Some(3),
            _ => None,
        },
        Key::Named(NamedKey::Space) => Some(4),
        _ => None,
    }
}

fn mouse_bit(button: MouseButton) -> Option<u32> {
    match button {
        MouseButton::Left => Some(0),
        MouseButton::Right => Some(1),
        MouseButton::Middle => Some(2),
        _ => None,
    }
}

impl ApplicationHandler for HarnessState {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let game: GameManifest = match dag_runner::load_game_manifest(&self.manifest_path) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("Error: {}", e);
                event_loop.exit();
                return;
            }
        };

        self.render_kernel_name = game
            .render_kernel
            .clone()
            .or_else(|| self.render_kernel_name.clone());

        let params = match crate::schema::resolve_params(&game.design_params) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Error: {}", e);
                event_loop.exit();
                return;
            }
        };

        self.world_width_fix16 = *params.raw.get("world_width")
            .expect("missing required design parameter: world_width") as i32;
        self.world_height_fix16 = *params.raw.get("world_height")
            .expect("missing required design parameter: world_height") as i32;
        self.max_delta_time_fix16 = to_fix16(*params.display.get("max_delta_time")
            .expect("missing required design parameter: max_delta_time")) as i32;

        let ww_u32 = *params.raw.get("display_width")
            .expect("missing required design parameter: display_width") as u32;
        let wh_u32 = *params.raw.get("display_height")
            .expect("missing required design parameter: display_height") as u32;
        assert!(ww_u32 % 64 == 0, "display_width must be a multiple of 64");
        assert!(wh_u32 % 64 == 0, "display_height must be a multiple of 64");

        let win_attrs = Window::default_attributes()
            .with_title("Forge Engine")
            .with_inner_size(winit::dpi::PhysicalSize::new(ww_u32, wh_u32))
            .with_resizable(false);
        let window =
            Arc::new(event_loop.create_window(win_attrs).expect("Failed to create window"));

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: self.backends,
            ..Default::default()
        });
        let surface = instance
            .create_surface(window.clone())
            .expect("Failed to create surface");
        let (adapter, device, queue) = gpu::init_device_for_surface(&instance, &surface);

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .find(|f| **f == wgpu::TextureFormat::Rgba8UnormSrgb)
            .or_else(|| caps.formats.first())
            .copied()
            .expect("No surface formats");

        surface.configure(
            &device,
            &wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::RENDER_ATTACHMENT,
                format,
                width: ww_u32,
                height: wh_u32,
                present_mode: wgpu::PresentMode::AutoVsync,
                desired_maximum_frame_latency: 2,
                alpha_mode: caps
                    .alpha_modes
                    .first()
                    .copied()
                    .unwrap_or(wgpu::CompositeAlphaMode::Auto),
                view_formats: vec![],
            },
        );

        let mut runner = match DagRunner::new(device, queue, &game, &self.registry_dir) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("Error: {}", e);
                event_loop.exit();
                return;
            }
        };

        if let Some(ref rk_name) = self.render_kernel_name {
            match registry::lookup_kernel(&self.registry_dir, rk_name) {
                Ok(Some(entry)) => {
                    if let Err(e) = runner.compile_kernel(&entry.contract) {
                        eprintln!("Error compiling render kernel: {}", e);
                        event_loop.exit();
                        return;
                    }
                }
                Ok(None) => {
                    eprintln!("Render kernel '{}' not in registry", rk_name);
                    event_loop.exit();
                    return;
                }
                Err(e) => {
                    eprintln!("Error: {}", e);
                    event_loop.exit();
                    return;
                }
            }
        }

        self.surface_width = ww_u32;
        self.surface_height = wh_u32;
        self.window = Some(window);
        self.surface = Some(surface);
        self.runner = Some(runner);
        self.last_frame = Instant::now();
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: ref key,
                        state,
                        ..
                    },
                ..
            } => {
                if state == ElementState::Pressed && *key == Key::Named(NamedKey::Escape) {
                    event_loop.exit();
                    return;
                }
                if let Some(bit) = key_bit(key) {
                    match state {
                        ElementState::Pressed => self.keys_down |= 1 << bit,
                        ElementState::Released => self.keys_down &= !(1 << bit),
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.mouse_x = position.x;
                self.mouse_y = position.y;
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if let Some(bit) = mouse_bit(button) {
                    match state {
                        ElementState::Pressed => self.mouse_buttons |= 1 << bit,
                        ElementState::Released => self.mouse_buttons &= !(1 << bit),
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                self.run_frame();
                if let Some(ref win) = self.window {
                    win.request_redraw();
                }
            }
            _ => {}
        }
    }
}

impl HarnessState {
    fn run_frame(&mut self) {
        let runner = match self.runner.as_mut() {
            Some(r) => r,
            None => return,
        };

        let now = Instant::now();
        let dt_secs = now.duration_since(self.last_frame).as_secs_f64();
        self.last_frame = now;
        let dt_clamped = (to_fix16(dt_secs) as i32).min(self.max_delta_time_fix16);

        self.frame_number += 1;
        let random_seed: u32 = self.rng.r#gen();

        let mouse_x_fix16 = to_fix16(self.mouse_x) as i32;
        let mouse_y_fix16 = to_fix16(self.mouse_y) as i32;

        runner.upload_buffer(
            "InputBuffer",
            &pack_input_buffer(self.keys_down, mouse_x_fix16, mouse_y_fix16, self.mouse_buttons),
        );
        runner.upload_buffer(
            "GlobalsBuffer",
            &pack_globals_buffer(
                self.frame_number,
                dt_clamped,
                self.world_width_fix16,
                self.world_height_fix16,
                random_seed,
            ),
        );

        match runner.run_frame(self.verify) {
            Ok(result) => {
                if !result.passed {
                    for e in &result.errors {
                        eprintln!("Frame {}: {}", result.frame, e);
                    }
                }
            }
            Err(e) => eprintln!("Frame error: {}", e),
        }

        // Render kernel dispatch + copy to surface
        if let Some(ref rk_name) = self.render_kernel_name.clone() {
            let surface = match self.surface.as_ref() {
                Some(s) => s,
                None => return,
            };
            let frame = match surface.get_current_texture() {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("Surface error: {}", e);
                    return;
                }
            };

            let runner = self.runner.as_ref().unwrap();
            let mut encoder =
                runner
                    .device()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("render_encoder"),
                    });

            if let Err(e) = runner.dispatch_kernel(&mut encoder, rk_name) {
                eprintln!("Render dispatch error: {}", e);
                frame.present();
                return;
            }

            // Copy Framebuffer GPU buffer → surface texture
            if let Ok(fb_buf) = runner.current_buffer("Framebuffer") {
                encoder.copy_buffer_to_texture(
                    wgpu::TexelCopyBufferInfo {
                        buffer: fb_buf,
                        layout: wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(self.surface_width * 4),
                            rows_per_image: Some(self.surface_height),
                        },
                    },
                    wgpu::TexelCopyTextureInfo {
                        texture: &frame.texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::Extent3d {
                        width: self.surface_width,
                        height: self.surface_height,
                        depth_or_array_layers: 1,
                    },
                );
            }

            runner.queue().submit(Some(encoder.finish()));
            frame.present();
        }
    }
}

pub fn run_harness(
    manifest_path: &Path,
    registry_dir: &Path,
    backends: Backends,
    verify: bool,
    seed: Option<u64>,
) {
    let rng = match seed {
        Some(s) => SmallRng::seed_from_u64(s),
        None => SmallRng::from_entropy(),
    };

    let mut state = HarnessState {
        keys_down: 0,
        mouse_x: 0.0,
        mouse_y: 0.0,
        mouse_buttons: 0,
        last_frame: Instant::now(),
        frame_number: 0,
        max_delta_time_fix16: 0,
        world_width_fix16: 0,
        world_height_fix16: 0,
        rng,
        manifest_path: manifest_path.to_path_buf(),
        registry_dir: registry_dir.to_path_buf(),
        backends,
        verify,
        render_kernel_name: None,
        window: None,
        surface: None,
        surface_width: 0,
        surface_height: 0,
        runner: None,
    };

    let event_loop = EventLoop::builder()
    .with_x11()
    .build()
    .expect("Failed to create event loop");
    
    event_loop.run_app(&mut state).expect("Event loop error");
}
