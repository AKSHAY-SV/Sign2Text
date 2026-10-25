# ✊ Real-time Sign Language Translation with Citizen ASL Dataset 

This project implements a full pipeline to translate sign language into text using pose estimation, deep learning, and a large language model (LLM) for sentence refinement. The live translation system is built using the [Citizen ASL Dataset](https://www.microsoft.com/en-us/research/project/asl-citizen/) and runs in real time with a webcam.

<p align="center">
  <a href="https://www.youtube.com/watch?v=gJ-PRa88E-M">
    <img src="demo/asldemo.gif" alt="Live ASL Translation Demo">
  </a>
</p>

Watch a real-time demo of sign language translation using our trained model and the Citizen ASL dataset.
> Disclaimer: Not a professional signer, the signing in this demo may not fully follow formal ASL grammar.

## Features
- Pose-based sign language recognition using MediaPipe Holistic
- Normalization and preprocessing of landmark sequences
- GRU-based model trained on 200 unique glosses (271 total classes including duplicates) (approx. 80% accuracy on validation)
- Real-time webcam-based translation with LLM for sentence refinement (OpenAI GPT)


## Setup

### Requirements
Make sure to have Python 3.10.17 installed. You can use pyenv or another version manager to install and activate it.

### Clone the repository
```bash
git clone https://github.com/tobp03/live-asl-translation.git
cd live-asl-translation
```
### Install Dependencies
Once your done, in the correct environment:
```bash
pip install -r requirements.txt
```

### Obtain API Key for LLM
This project uses [OpenAI GPT API](https://platform.openai.com) for sentence refinement during live translation. You need a valid API key with billing enabled. Crete a file named `.env` in the project root and add the following line:
```bash
API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

## 🚀 Usage

### Integrated laptop and LPC2148 application

From the repository root, start the camera, both existing recognizers, laptop
text display, and (when connected) LCD output in one command:

```sh
python3 tools/run_sign2text_integrated.py --serial-port /dev/ttyUSB0
```

Replace `/dev/ttyUSB0` with the actual USB-serial device (for example
`/dev/ttyACM0` or `COM4`). The LCD receiver uses UART0 at 9600 baud, 8N1, no
flow control; Python defaults to the same baud. Use `--serial-baud` only when
the firmware is configured to match. Without `--serial-port`, laptop
recognition runs without hardware output. A missing/unavailable serial port
prints a warning and leaves laptop recognition running.

The firmware sources are in `embedded/lpc2148_keil`. Add them to the existing
lab Keil project, set `UART0_PCLK_HZ` from the project's actual peripheral
clock, build the HEX, and flash it once using the lab procedure before using
LCD output. No Keil project or board verification is included in this
repository. Check the detailed wiring and clock assumptions in
[`embedded/lpc2148_keil/README.md`](embedded/lpc2148_keil/README.md).

- Full pipeline
   To train your own model and reproduce the full workflow : Open and run `main_livefeed.ipynb`. This notebook contains all steps from start to finish:
   1.  Download the [Citizen ASL Dataset](https://www.microsoft.com/en-us/research/project/asl-citizen/)
   2.  Place the dataset folder in the same directory as `main_livefeed.ipynb` (the project root)
   3.  Open and run the full pipeline step 1-3 on `main_livefeed.ipynb`
   4.  Finally, use the trained model for live translation
- Direct use with Pre-trained model
  If you prefer to skip training
  1. Ensure the pre-trained GRU model (`.keras`) is available in the `/models` folder (already provided)
  2. Open `main_livefeed.ipynb`
  3. Jump directly to **Step 4** to start live translation using the pre-trained model
- Preprocessed data and model training is also available on [Kaggle](https://www.kaggle.com/code/tobypu/aslcitizen-top200-training). 

## References
- **ASL Citizen Dataset**  
  Desai, A., Berger, L., Minakov, F. O., Milan, V., Singh, C., Pumphrey, K., Ladner, R. E., Daumé III, H., Lu, A. X., Caselli, N., & Bragg, D.  
  *ASL Citizen: A Community-Sourced Dataset for Advancing Isolated Sign Language Recognition*.  
  arXiv preprint arXiv:2304.05934, 2023. [[Paper](https://arxiv.org/abs/2304.05934)]
  
- **Keypoint Preprocessing** <br>
  Roh, K., Lee, H., Hwang, E. J., Cho, S., & Park, J. C. (2024, May). Preprocessing Mediapipe Keypoints with Keypoint Reconstruction and Anchors for Isolated Sign Language Recognition. In *Proceedings of the LREC-COLING   2024 11th Workshop on the Representation and Processing of Sign Languages: Evaluation of Sign Language Resources* (pp. 323–334). [[Paper](https://aclanthology.org/2024.signlang-1.36.pdf)]
  
## Related Publication
This project is related to the research paper:  
Purbojo, T., & Wijaya, A. (2025). Enhancing Pose-Based Sign Language Recognition: A Comparative Study of Preprocessing Strategies with GRU and LSTM. *Advance Sustainable Science Engineering and Technology*, 7(2), 02502017-02502017. [[Paper](https://journal2.upgris.ac.id/index.php/asset/article/view/1658)]

