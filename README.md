# **Trajectory Anomaly Detection with By-Design Complementary Detectors**

This repository demonstrates the pipeline for CETrajAD. The method leverages:
- **Data Preprocessing**: Process raw trajectory datasets into 3 types, including Speed, Route, and Shape.
- **Encoder Training**: Training LSTM autoencoders for each type.
- **Trajectory Encoding**: Encoding trajectories to generate embeddings and reconstruction losses.
- **Complementary Ensemble**: Combining outputs of complementary detectors to enhance anomaly detection performance.

---

## **Installation**
Clone this repository:
   ```bash
   git clone https://github.com/your-repo-name.git
   cd your-repo-name
   bapip install -r requirements.txt
   ```


## Citation
If you find this work helpful, please consider citing our paper:
@inproceedings{CETrajAD,
  title={Trajectory Anomaly Detection with By-Design Complementary Detectors},
  author={Cao, Shurui and Akoglu, Leman},
  booktitle={2025 SIAM International Conference on Data Mining (SDM25)},
  pages={},
  year={2025},
  organization={SIAM}
}


