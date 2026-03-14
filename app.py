import uvicorn
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import pickle

app = FastAPI()

# Load the model
with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    Age: int = Form(...),
    Experience: int = Form(...),
    Income: int = Form(...),
    Family: int = Form(...),
    CCAvg: float = Form(...),
    Education: int = Form(...),
    Mortgage: int = Form(...),
    Securities_Account: int = Form(..., alias="Securities Account"),
    CD_Account: int = Form(..., alias="CD Account"),
    Online: int = Form(...),
    CreditCard: int = Form(...)
):
    input_data = pd.DataFrame({
        'Age': [Age],
        'Experience': [Experience],
        'Income': [Income],
        'Family': [Family],
        'CCAvg': [CCAvg],
        'Education': [Education],
        'Mortgage': [Mortgage],
        'Securities Account': [Securities_Account],
        'CD Account': [CD_Account],
        'Online': [Online],
        'CreditCard': [CreditCard]
    })

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        prediction_text = "Congratulations! Your loan is likely to be Approved."
    else:
        prediction_text = "We're sorry, but your loan is likely to be Rejected."

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "prediction_text": prediction_text}
    )

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
