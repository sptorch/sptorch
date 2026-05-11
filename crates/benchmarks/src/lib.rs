//! SPTorch 内部性能基线 crate。
//!
//! 这里不暴露生产 API，只承载 Criterion benchmark。把 benchmark 独立成 workspace
//! 成员，可以让 CI 检查性能入口是否持续可运行，同时避免污染可发布 crate 的依赖树。
