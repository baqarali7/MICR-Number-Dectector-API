import base64
import numpy as np
import io as io
from io import StringIO
from PIL import Image
from skimage.segmentation import clear_border
from imutils import contours
import numpy as np
import imutils
import cv2
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin

app = Flask(__name__, template_folder="templates")

cors = CORS(app)

def get_ref_img():
    ref = cv2.imread('C:/Users/pc/Desktop/API2/00.png')
    print("** Reference Image Loaded **")
    return ref
    

def extract_digits_and_symbols(image, charCnts, minW=5, minH=15):
    charIter = charCnts.__iter__()
    rois = []
    locs = []
    while True:
        try:
            c = next(charIter)
            (cX, cY, cW, cH) = cv2.boundingRect(c)
            roi = None
            if cW >= minW and cH >= minH:
                # extract the ROI
                roi = image[cY:cY + cH, cX:cX + cW]
                rois.append(roi)
                locs.append((cX, cY, cX + cW, cY + cH))
            else:
                parts = [c, next(charIter), next(charIter)]
                (sXA, sYA, sXB, sYB) = (np.inf, np.inf, -np.inf, -np.inf)
                for p in parts:
                    (pX, pY, pW, pH) = cv2.boundingRect(p)
                    sXA = min(sXA, pX)
                    sYA = min(sYA, pY)
                    sXB = max(sXB, pX + pW)
                    sYB = max(sYB, pY + pH)
                # extract the ROI
                roi = image[sYA:sYB, sXA:sXB]
                rois.append(roi)
                locs.append((sXA, sYA, sXB, sYB))
        except StopIteration:
            break
    return (rois, locs)

def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert("RGB")
    image = img_to_array(image)
    return image


@app.route('/')
def home():
    return render_template('Predict.html')

print("** Loading Reference Image **")
get_ref_img()


@app.route('/fetch', methods=['POST'])
@cross_origin(origin='localhost',headers=['Content- Type','Authorization'])
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = io.BytesIO(decoded)
    image = cv2.imdecode(np.fromstring(image.read(), np.uint8), 1)

    charNames = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
        "T", "U", "A", "D"]
    print("** Image Translated **")
    ref = get_ref_img()
    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    ref = imutils.resize(ref, width=400)
    ref = cv2.threshold(ref, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    refCnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    refCnts = imutils.grab_contours(refCnts)
    refCnts = contours.sort_contours(refCnts, method="left-to-right")[0]
    refROIs = extract_digits_and_symbols(ref, refCnts,
        minW=10, minH=20)[0]
    chars = {}
    for (name, roi) in zip(charNames, refROIs):
        roi = cv2.resize(roi, (36, 36)) 
        chars[name] = roi
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 7))
    output = []
    (h, w,) = image.shape[:2]
    delta = int(h - (h * 0.2))
    bottom = image[delta:h, 0:w]
    gray = cv2.cvtColor(bottom, cv2.COLOR_BGR2GRAY)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
    gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0,
        ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
    gradX = gradX.astype("uint8")
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
    thresh = cv2.threshold(gradX, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thresh = clear_border(thresh)
    groupCnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    groupCnts = imutils.grab_contours(groupCnts)
    groupLocs = []
    for (i, c) in enumerate(groupCnts):
        (x, y, w, h) = cv2.boundingRect(c)
        if w > 50 and h > 15:
            groupLocs.append((x, y, w, h))
    groupLocs = sorted(groupLocs, key=lambda x:x[0])
    for (gX, gY, gW, gH) in groupLocs:
        # initialize the group output of characters
        groupOutput = []
        group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
        group = cv2.threshold(group, 0, 255,
            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        charCnts = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        charCnts = imutils.grab_contours(charCnts)
        charCnts = contours.sort_contours(charCnts,
            method="left-to-right")[0]
        (rois, locs) = extract_digits_and_symbols(group, charCnts)
        for roi in rois:
            scores = []
            roi = cv2.resize(roi, (36, 36))
            for charName in charNames:
                result = cv2.matchTemplate(roi, chars[charName],
                    cv2.TM_CCOEFF)
                (_, score, _, _) = cv2.minMaxLoc(result)
                scores.append(score)
            groupOutput.append(charNames[np.argmax(scores)])
        cv2.rectangle(image, (gX - 10, gY + delta - 10),
            (gX + gW + 10, gY + gY + delta), (0, 0, 255), 2)
        cv2.putText(image, "".join(groupOutput),
            (gX - 10, gY + delta - 25), cv2.FONT_HERSHEY_SIMPLEX,
            0.95, (0, 0, 255), 3)
        output.append("".join(groupOutput))
        print("** MICR CODE FETCHED **")
        print(output)
        response = {
            'prediction' : {
                'Result' : output,
            }
        }
        return jsonify(response)