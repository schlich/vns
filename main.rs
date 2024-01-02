fn main() {
    use plotly::{Plot, Scatter};

    let mut plot = Plot::new();
    let trace = Scatter::new(vec![0, 1, 2], vec![2, 1, 0]);
    plot.add_trace(trace);

    plot.write_html("out.html");
}