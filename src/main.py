import io

import uvicorn
import cv2

from starlette.responses import StreamingResponse, FileResponse
from fastapi import UploadFile, File, FastAPI, Form
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi import APIRouter, HTTPException, File, UploadFile, Header
from application.fetch_model.services.fetch_model_service import FetchModelService
from application.inference import inference
from application.fetch_labels.fetch_labels import FetchLabels
from application.fetch_configuration.fetch_configuration import FetchConfiguration
from application.fetch_palette.fetch_palette import FetchPalette
from application.inference_return_json import inference_json
from application.inference_for_anonymization import inference_anon
from models import ApiResponse


app = FastAPI(version='1.0', title='BMW InnovationLab OpenVINO Segmentation inference',
			  description="<b>API for performing OpenVINO Segmentation inference</b></br></br>"
						  "<b>Contact the developers:</b></br>"
						  "<b>Elio Hanna: <a href='mailto:elio.hanna@bmw.de'>elio.hanna@bmw.de</a></b></br>"
						  "<b>BMW Innovation Lab: <a href='mailto:innovation-lab@bmw.de'>innovation-lab@bmw.de</a></b>")


@app.get("/load")
async def load_models():
    """
    Lists all available models
    :return: names with uuid
    """
    try:
        return FetchModelService().fetch_all_models()
    except Exception:
        raise HTTPException(status_code=300, detail="cannot list all the models")


@app.get('/models')
async def list_models(user_agent: str = Header(None)):
    """
    Lists all available models.
    :param user_agent:
    :return: APIResponse
    """
    return ApiResponse(data={'models': FetchModelService().fetch_all_models()})


@app.get('/models/{model_name}/labels')
async def list_model_labels(model_name: str):
    """
	Lists all the model's labels.
	:param model_name: Model name
	:return: List of model's labels
	"""
    try:
        labels = FetchLabels().get_labels(model_name)
        return ApiResponse(data=labels)
    except Exception:
        raise HTTPException(status_code=404, detail="something went wrong! please check if the model exist")


@app.get('/models/{model_name}/config')
async def list_model_configuration(model_name: str):
    """
	Lists all the model's configuration.
	:param model_name: Model name
	:return: List of model's configuration
	"""
    try:
        configuration = FetchConfiguration().get_configuration(model_name)
        return ApiResponse(data=configuration)
    except Exception:
        raise HTTPException(status_code=404, detail="something went wrong! please check if the model exist")


@app.get('/models/{model_name}/palette')
async def list_model_palette(model_name: str):
    """
	Lists all the model's palette.
	:param model_name: Model name
	:return: List of model's palette
	"""
    try:
        palette = FetchPalette().get_palette(model_name)
        return ApiResponse(data=palette)
    except Exception:
        raise HTTPException(status_code=404, detail="something went wrong! please check if the model exist")


@app.post('/models/{model_name}/image_segmentation')
async def image_segmentation(model_name: str, input_data: UploadFile = File(...)):
    """
    Segment the Image and return the result as image
    :param model_name: Model name
    :param input_data: Image file
    :return: Image file
    """
    try:
        result = inference.Inference()
        final_image = result.image_inference(model_name=model_name, input_data=input_data)
        res, im_png = cv2.imencode(".png", final_image)
        return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")
    except Exception:
        raise HTTPException(status_code=300, detail="unexpected server error")


@app.post('/models/{model_name}/inference')
async def image_inference(model_name: str, input_data: UploadFile = File(...)):
    """
    Segment the Image and return the result as image
    :param model_name: Model name
    :param input_data: Image filec
    :return: Image file
    """
    try:
        result = inference_anon.Inference()
        result.image_inference(model_name=model_name, input_data=input_data)
        return FileResponse("result.jpg", media_type="image/jpg")
    except Exception:
        raise HTTPException(status_code=300, detail="unexpected server error")


@app.post('/models/{model_name}/detect')
async def image_segmentation_json(model_name: str, input_data: UploadFile = File(...)):
    """
    Segment the Image and return the result as json
    :param model_name: Model name
    :param input_data: Image file
    :return: Json Object
    """
    try:
        result = inference_json.Inference()
        json_result = result.image_inference(model_name=model_name, input_data=input_data)
        json_compatible_item_data = jsonable_encoder(json_result)
        return JSONResponse(content=json_compatible_item_data)
    except Exception:
        raise HTTPException(status_code=300, detail="unexpected server error")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
