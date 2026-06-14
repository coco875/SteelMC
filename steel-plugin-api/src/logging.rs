use crate::hook::{HookApiVtableDyn, HookApiVtableRef};

#[stabby::stabby]
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    Error = 1,
    Warn = 2,
    Info = 3,
    Debug = 4,
    Trace = 5,
}

impl From<log::Level> for LogLevel {
    fn from(level: log::Level) -> Self {
        match level {
            log::Level::Error => LogLevel::Error,
            log::Level::Warn => LogLevel::Warn,
            log::Level::Info => LogLevel::Info,
            log::Level::Debug => LogLevel::Debug,
            log::Level::Trace => LogLevel::Trace,
        }
    }
}

impl From<LogLevel> for log::Level {
    fn from(level: LogLevel) -> Self {
        match level {
            LogLevel::Error => log::Level::Error,
            LogLevel::Warn => log::Level::Warn,
            LogLevel::Info => log::Level::Info,
            LogLevel::Debug => log::Level::Debug,
            LogLevel::Trace => log::Level::Trace,
        }
    }
}

struct FfiLogger {
    host_api: HookApiVtableRef,
}

impl log::Log for FfiLogger {
    fn enabled(&self, _metadata: &log::Metadata) -> bool {
        true
    }

    fn log(&self, record: &log::Record) {
        if self.enabled(record.metadata()) {
            let level = LogLevel::from(record.level());
            let target = crate::AbiString::from(record.target());
            let message = crate::AbiString::from(format!("{}", record.args()).as_str());
            self.host_api.log(level, target, message);
        }
    }

    fn flush(&self) {}
}

static LOGGER_INIT: std::sync::Once = std::sync::Once::new();

/// Initialize the global logger for the plugin, forwarding all log messages to the host.
pub fn init_logger(host_api: HookApiVtableRef) {
    LOGGER_INIT.call_once(|| {
        let logger = Box::leak(Box::new(FfiLogger { host_api }));
        let _ = log::set_logger(logger);
        log::set_max_level(log::LevelFilter::Trace);
    });
}
