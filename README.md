# Facial Age Estimator API

FastAPI service that predicts age from a face photo, using a MobileNet CNN fine-tuned on ~184k images from [IMDB-Clean](https://www.kaggle.com/datasets/yuulind/imdb-clean). **Test MAE: 5.93 years.**

## Usage

```bash
docker build -t age-api .
docker run -p 8000:8000 age-api
```

```bash
curl -X POST http://localhost:8000/predict -F "file=@face.jpg"
# {"predicted_age": 31.4}
```

Docs: `http://localhost:8000/docs` · Health check: `GET /health`

## Model

MobileNet (ImageNet weights) + `Dense(1)` head, fully fine-tuned with Adam and MAE loss. Training code: `age_prediction.ipynb`.

## Limitations

No face detection (send a close-up of one face). Trained on celebrity photos, so accuracy may drop on everyday images.

**Stack:** Python · TensorFlow/Keras · FastAPI · Docker
