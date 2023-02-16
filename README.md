# Flashnet: Towards Reproducible Data Science Platform for Storage Systems

In the ML/Deep Learning community, the large ImageNet benchmarks have spurred research in image recognition. Similarly, we would like to provide benchmarks for fostering storage research in ML-based per-IO latency prediction. Therefore, we present FlashNet, a reproducible data science platform for storage systems. To start a big task, we use I/O latency prediction as a case study. Thus, FlashNet has been built for I/O latency prediction tasks. With FlashNet, data engineers can collect the IO traces of various devices. The data scientists then can train the ML models to predict the IO latency based on those traces. All traces, results, and codes will be shared in the FlashNet training ground platform which utilizes Chameleon trovi for better reproducibility.

## Files Explanation

There are 3 main files to run,

1. readme.ipynb <br> Run this Jupyter Notebook to run FlashNet on Chameleon Trovi's reserved machine.
2. readme-remoteserver.ipynb <br> Run this Jupyter Notebook to run FlashNet on pre-reserved Chameleon Cloud machine through Chameleon Trovi.
3. readme-remoteserver-chi.ipynb <br> Run this Jupyter Notebook to run FlashNet by reserving a node on Chameleon Cloud through Chameleon Trovi.
