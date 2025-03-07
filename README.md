# **Audio Drone Classification**  

*I conducted this project during my final year at ENSTA Bretagne as a engineering student*
*This project explores methods that combine time-frequency representations and deep learning for audio-based drone classification.*

## 🚀 **Introduction**  
This project utilizes an audio drone dataset provided by [https://ieeexplore.ieee.org/document/8766732]. The primary contributions include:
- An adaptable pipeline for studying drone detection and classification using various time-frequency representations.
- A structured approach to data preprocessing, model training, and evaluation.

The work conducted in this project is further detailed in the associated paper.

## 🎯 **Project Structure**  
- **`data/`**: Contains the dataset from the reference paper (with unknown samples removed to avoid class dominance).
- **`src/`**: Includes all core functionalities such as data preprocessing, model implementation, training, and evaluation.
- **`dataset_analysis.ipynb`**: Use this notebook to explore the dataset.
- **`detection.ipynb`**: Focuses on drone detection.
- **`classification.ipynb`**: Focuses on drone classification.

## 🛠 **Setup**  
1. **Clone the repository**  
   ```bash
   git clone [https://github.com/RaphaELGrisel/Drone_Audio_Classification].git
   cd nom-du-projet
   ```  
2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt  
   ```  
3. **Run the notebooks**  
   Open and execute the cells in one of the Jupyter notebooks.

4. **Pretrained Models**  
   Pretrained CNN models for detection and classification can be found in the `saved_model/` directory.

## 📢 **Additional Information**  
For more details, refer to the linked paper and explore the provided notebooks.
