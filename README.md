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

### Laptop application

From the repository root, start the webcam-based word and alphabet recognizers
with the laptop preview and text display:

```sh
./run_sign2text_integrated.sh
```

Use `1` for WORD mode and `2` for SPELL mode. The standalone recognition
commands remain available as `./run_sign2text_words.sh` and
`./run_sign2text_alphabet.sh`.

ESP32 + ILI9341 TFT output is enabled by default by the integrated launcher.
The laptop sends the accepted text buffer over `/dev/ttyUSB0` at 115200 baud;
the ESP32 firmware should already be uploaded and listening for newline-ended
UTF-8 messages. The sender uses a background worker, reports connection errors,
and retries without stopping camera recognition. Disable serial output with
`./run_sign2text_integrated.sh --no-tft`, or select another port/baud with
`--tft-port` and `--tft-baud`. Text is sent when the accepted buffer changes,
not on every prediction frame.

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

