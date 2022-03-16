import streamlit as st
import matplotlib.pyplot as plt
from ising import ising

from generate_data import generate_data


def main():
    st.set_page_config(
        page_title="Ising Model Resource Allocation Demo",
    )

    with st.sidebar.form(key="Data Generator"):
        st.write("## Data Generator")
        data_size = st.slider("Data size", min_value=1, max_value=100, value=10)

        cluster_std = st.slider(
            "Standard Deviation of Clusters", min_value=0.0, max_value=10.0, value=1.0
        )
        generate_button = st.form_submit_button(label="Generate")

    with st.sidebar.form(key="Ising Model Resource Allocation Demo"):
        st.write("## Options")
        max_iterations = st.slider(
            "Maximum iterations", min_value=500, max_value=10000, value=1000
        )
        local_threshold = st.slider(
            "Local threshold", min_value=10, max_value=30, value=10
        )

        draw_every_n_iterations = st.slider(
            "Draw every n iterations", min_value=1, max_value=500, value=5
        )
        train_button = st.form_submit_button(label="Train")

    st.title(
        "Ising Model Resource Allocation Demo",
    )

    if "fig" not in st.session_state:
        st.session_state.fig, st.session_state.ax = plt.subplots(figsize=(14, 12))
        st.session_state.ax.axes.xaxis.set_visible(False)
        st.session_state.ax.axes.yaxis.set_visible(False)
        st.session_state.data_plot = st.pyplot(st.session_state.fig)
        st.session_state.res_box = st.empty()
    else:
        st.session_state.ax.clear()
        st.session_state.res_box.empty()

    if generate_button:
        (
            st.session_state.sample_data,
            st.session_state.sample_data_labels,
        ) = generate_data(cluster_std, data_size)

        st.session_state.ax.scatter(
            st.session_state.sample_data[:, 0],
            st.session_state.sample_data[:, 1],
            c=["r" if x == 1 else "b" for x in st.session_state.sample_data_labels],
            alpha=0.5,
        )
        st.session_state.data_plot.pyplot(st.session_state.fig)

    if train_button:
        if "sample_data" not in st.session_state:
            st.error("You must generate data first.")
            return

        ising = ising(
            st.session_state.sample_data,
            st.session_state.sample_data_labels,
            maxiter=max_iterations,
            local_thresh=local_threshold,
            draw_every_n_iterations=draw_every_n_iterations,
            data_plot=st.session_state.data_plot,
            fig=st.session_state.fig,
            ax=st.session_state.ax,
            c=st.session_state.res_box,
        )
        print(ising)


if __name__ == "__main__":
    main()
