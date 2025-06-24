use criterion::{black_box, criterion_group, criterion_main, Criterion};
use arrow_graph::prelude::*;

fn benchmark_shortest_path(c: &mut Criterion) {
    c.bench_function("shortest_path", |b| {
        b.iter(|| {
            // TODO: Implement actual benchmark
            black_box(42)
        })
    });
}

fn benchmark_pagerank(c: &mut Criterion) {
    c.bench_function("pagerank", |b| {
        b.iter(|| {
            // TODO: Implement actual benchmark  
            black_box(42)
        })
    });
}

criterion_group!(benches, benchmark_shortest_path, benchmark_pagerank);
criterion_main!(benches);