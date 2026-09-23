import cv2
import numpy as np
import time
import ai_edge_litert.interpreter as tflite


# =========================================================
# 1. MODEL
# =========================================================

modelPath = "best.tflite"

interpreter = tflite.Interpreter(model_path=modelPath)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]["shape"]

print("Input shape :", input_shape)
print("Output shape:", output_details[0]["shape"])
print("Input dtype :", input_details[0]["dtype"])
print("Output dtype:", output_details[0]["dtype"])


# =========================================================
# 2. PRODUCT INFORMATION
# =========================================================

product_name = {
    0: "Coffee",
    1: "Water",
    2: "Letbe",
    3: "Cola",
    4: "Pocachip",
    5: "Yukgaejang",
    6: "Khancho",
    7: "Pepero"
}

product_price = {
    0: 2500,
    1: 1100,
    2: 1000,
    3: 1700,
    4: 2100,
    5: 1200,
    6: 1600,
    7: 2000
}


# =========================================================
# 3. YOLO SETTINGS
# =========================================================

IMG_SIZE = 320
CONF_TH = 0.4
STABLE_TIME = 2.0


# =========================================================
# 4. CART / STATE
# =========================================================

cart = []

candidate_product = None
candidate_start_time = 0

last_added_product = None

cart_message = ""
cart_message_time = 0
CART_MESSAGE_DURATION = 1.5

payment_mode = False
payment_complete = False
paid_total = 0


# =========================================================
# 5. LETTERBOX
# =========================================================

def letterbox(image, new_shape=(320, 320)):

    shape = image.shape[:2]

    ratio = min(
        new_shape[0] / shape[0],
        new_shape[1] / shape[1]
    )

    new_unpad = (
        int(round(shape[1] * ratio)),
        int(round(shape[0] * ratio))
    )

    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]

    dw /= 2
    dh /= 2

    resized = cv2.resize(
        image,
        new_unpad,
        interpolation=cv2.INTER_LINEAR
    )

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    canvas = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114)
    )

    return canvas, ratio, dw, dh


# =========================================================
# 6. IMAGE PROCESSING
# =========================================================

def processImage(frame):

    image, ratio, pad_x, pad_y = letterbox(
        frame,
        (IMG_SIZE, IMG_SIZE)
    )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = image.astype(np.float32) / 255.0

    expected_shape = input_details[0]["shape"]

    # -----------------------------------------------------
    # NCHW : [1, 3, 320, 320]
    # -----------------------------------------------------

    if len(expected_shape) == 4 and expected_shape[1] == 3:

        image = np.transpose(
            image,
            (2, 0, 1)
        )

        image = np.expand_dims(
            image,
            axis=0
        )

    # -----------------------------------------------------
    # NHWC : [1, 320, 320, 3]
    # -----------------------------------------------------

    elif len(expected_shape) == 4 and expected_shape[3] == 3:

        image = np.expand_dims(
            image,
            axis=0
        )

    else:

        raise ValueError(
            f"Unsupported input shape: {expected_shape}"
        )

    # -----------------------------------------------------
    # Input dtype 처리
    # -----------------------------------------------------

    input_dtype = input_details[0]["dtype"]

    if input_dtype == np.float32:

        input_data = image.astype(np.float32)

    elif input_dtype == np.uint8:

        input_data = (image * 255).astype(np.uint8)

    elif input_dtype == np.int8:

        input_data = (
            image * 127
        ).astype(np.int8)

    else:

        input_data = image.astype(input_dtype)

    # -----------------------------------------------------
    # Tensor 입력
    # -----------------------------------------------------

    interpreter.set_tensor(
        input_details[0]["index"],
        input_data
    )

    interpreter.invoke()

    output = interpreter.get_tensor(
        output_details[0]["index"]
    )

    return output, ratio, pad_x, pad_y


# =========================================================
# 7. DETECTION RESULT
# =========================================================

def getBestProduct(output):

    data = np.squeeze(output)

    if data.ndim != 2:
        return None, None, 0

    # -----------------------------------------------------
    # 출력 형태 정리
    # -----------------------------------------------------

    if data.shape[0] < data.shape[1]:

        data = data.T

    best_class = None
    best_box = None
    best_conf = 0

    # -----------------------------------------------------
    # 각각의 Detection 확인
    # -----------------------------------------------------

    for row in data:

        if len(row) < 5:
            continue

        x = row[0]
        y = row[1]
        w = row[2]
        h = row[3]

        class_scores = row[4:]

        class_id = int(
            np.argmax(class_scores)
        )

        confidence = float(
            class_scores[class_id]
        )

        if confidence > best_conf:

            best_conf = confidence
            best_class = class_id

            best_box = (
                float(x),
                float(y),
                float(w),
                float(h)
            )

    # -----------------------------------------------------
    # Confidence threshold
    # -----------------------------------------------------

    if best_conf < CONF_TH:

        return None, None, 0

    return (
        best_class,
        best_box,
        best_conf
    )


# =========================================================
# 8. TOTAL PRICE
# =========================================================

def getTotal():

    total = 0

    for product in cart:

        total += product_price.get(
            product,
            0
        )

    return total


# =========================================================
# 9. RESET CART
# =========================================================

def resetCart():

    global cart
    global candidate_product
    global candidate_start_time
    global last_added_product
    global cart_message
    global cart_message_time

    cart = []

    candidate_product = None
    candidate_start_time = 0

    last_added_product = None

    cart_message = ""
    cart_message_time = 0


# =========================================================
# 10. DRAW BOUNDING BOX
# =========================================================

def drawBoundingBox(
    frame,
    detected_box,
    current_product,
    confidence,
    ratio,
    pad_x,
    pad_y
):

    if detected_box is None or current_product is None:
        return

    x, y, w, h = detected_box

    # 모델 좌표 → 카메라 좌표
    box_x1 = int((x - w / 2 - pad_x) / ratio)
    box_y1 = int((y - h / 2 - pad_y) / ratio)

    box_x2 = int((x + w / 2 - pad_x) / ratio)
    box_y2 = int((y + h / 2 - pad_y) / ratio)

    # 화면 범위 제한
    box_x1 = max(0, min(frame.shape[1] - 1, box_x1))
    box_y1 = max(0, min(frame.shape[0] - 1, box_y1))
    box_x2 = max(0, min(frame.shape[1] - 1, box_x2))
    box_y2 = max(0, min(frame.shape[0] - 1, box_y2))

    # ==========================================
    # Bounding Box
    # ==========================================

    cv2.rectangle(
        frame,
        (box_x1, box_y1),
        (box_x2, box_y2),
        (255, 255, 255),
        2
    )

    # ==========================================
    # 상품명 / 인식률
    # ==========================================

    name = product_name.get(
        current_product,
        "Unknown"
    )

    label = f"{name}  {confidence * 100:.1f}%"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    thickness = 2

    (text_w, text_h), _ = cv2.getTextSize(
        label,
        font,
        font_scale,
        thickness
    )

    # 라벨 위치
    label_x = box_x1

    if box_y1 > text_h + 15:
        label_y = box_y1
    else:
        label_y = box_y1 + text_h + 15

    # ==========================================
    # 반투명 라벨 배경
    # ==========================================

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (label_x, label_y - text_h - 14),
        (label_x + text_w + 20, label_y + 5),
        (25, 25, 25),
        -1
    )

    cv2.addWeighted(
        overlay,
        0.85,
        frame,
        0.15,
        0,
        frame
    )

    # ==========================================
    # 상품명 - 흰색
    # ==========================================

    cv2.putText(
        frame,
        name,
        (label_x + 8, label_y - 5),
        font,
        font_scale,
        (255, 255, 255),
        thickness
    )

    # 인식률은 노란색
    name_width = cv2.getTextSize(
        name,
        font,
        font_scale,
        thickness
    )[0][0]

    cv2.putText(
        frame,
        f"{confidence * 100:.1f}%",
        (
            label_x + name_width + 18,
            label_y - 5
        ),
        font,
        font_scale,
        (0, 220, 255),
        thickness
    )

# =========================================================
# 11. ITEM ADDED MESSAGE
# =========================================================
def drawCartMessage(frame):

    if time.time() - cart_message_time >= CART_MESSAGE_DURATION:
        return

    # ==========================================
    # 화면 중앙 위치
    # ==========================================

    frame_h, frame_w = frame.shape[:2]

    box_w = 420
    box_h = 130

    x1 = (frame_w - box_w) // 2
    y1 = (frame_h - box_h) // 2

    x2 = x1 + box_w
    y2 = y1 + box_h

    # ==========================================
    # 반투명 배경
    # ==========================================

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (x1, y1),
        (x2, y2),
        (25, 25, 25),
        -1
    )

    cv2.addWeighted(
        overlay,
        0.90,
        frame,
        0.10,
        0,
        frame
    )

    # ==========================================
    # ITEM ADDED
    # ==========================================

    title = "ITEM ADDED"

    font = cv2.FONT_HERSHEY_SIMPLEX
    title_scale = 0.85
    title_thickness = 2

    (title_w, title_h), _ = cv2.getTextSize(
        title,
        font,
        title_scale,
        title_thickness
    )

    title_x = x1 + (box_w - title_w) // 2
    title_y = y1 + 50

    cv2.putText(
        frame,
        title,
        (title_x, title_y),
        font,
        title_scale,
        (255, 255, 255),
        title_thickness
    )

    # ==========================================
    # 상품명 + 가격
    # ==========================================

    name = product_name.get(
        last_added_product,
        "Unknown"
    )

    price = product_price.get(
        last_added_product,
        0
    )

    product_text = f"{name} - W{price:,}"

    product_scale = 0.65
    product_thickness = 2

    (product_w, product_h), _ = cv2.getTextSize(
        product_text,
        font,
        product_scale,
        product_thickness
    )

    product_x = x1 + (box_w - product_w) // 2
    product_y = y1 + 95

    cv2.putText(
        frame,
        product_text,
        (product_x, product_y),
        font,
        product_scale,
        (0, 220, 255),
        product_thickness
    )


# =========================================================
# 12. CART PANEL
# =========================================================

def drawCartPanel(frame):

    panel_width = 320

    panel = np.zeros(
        (
            frame.shape[0],
            panel_width,
            3
        ),
        dtype=np.uint8
    )

    panel[:] = (
        35,
        35,
        35
    )

    # -----------------------------------------------------
    # TITLE
    # -----------------------------------------------------

    cv2.putText(
        panel,
        "SHOPPING CART",
        (25, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.line(
        panel,
        (20, 55),
        (300, 55),
        (100, 100, 100),
        1
    )

    # -----------------------------------------------------
    # CART ITEMS
    # -----------------------------------------------------

    y = 90

    counts = {}

    for product in cart:

        if product not in counts:
            counts[product] = 0

        counts[product] += 1

    for product, count in counts.items():

        name = product_name.get(
            product,
            "Unknown"
        )

        price = product_price.get(
            product,
            0
        )

        item_total = price * count

        text = f"{name} x{count}"

        cv2.putText(
            panel,
            text,
            (25, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (230, 230, 230),
            1
        )

        cv2.putText(
            panel,
            f"W{item_total:,}",
            (205, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (230, 230, 230),
            1
        )

        y += 35

    # -----------------------------------------------------
    # TOTAL
    # -----------------------------------------------------

    cv2.line(
        panel,
        (20, y + 5),
        (300, y + 5),
        (100, 100, 100),
        1
    )

    total = getTotal()

    cv2.putText(
        panel,
        "TOTAL",
        (25, y + 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        panel,
        f"W{total:,}",
        (170, y + 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 220, 100),
        2
    )

    # -----------------------------------------------------
    # CONTROLS
    # -----------------------------------------------------

    cv2.putText(
        panel,
        "ENTER : PAYMENT",
        (25, 390),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 180, 180),
        1
    )

    cv2.putText(
        panel,
        "R : RESET",
        (25, 420),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 180, 180),
        1
    )

    cv2.putText(
        panel,
        "Q : QUIT",
        (25, 450),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 180, 180),
        1
    )

    # -----------------------------------------------------
    # CAMERA + CART
    # -----------------------------------------------------

    result = np.hstack(
        (
            frame,
            panel
        )
    )

    return result


# =========================================================
# 13. PAYMENT SCREEN
# =========================================================

def drawPaymentScreen(frame):

    screen = np.zeros_like(frame)

    screen[:] = (
        245,
        245,
        245
    )

    cv2.putText(
        screen,
        "PAYMENT",
        (210, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (30, 30, 30),
        3
    )

    total = getTotal()

    cv2.putText(
        screen,
        f"TOTAL  W{total:,}",
        (170, 190),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (30, 30, 30),
        2
    )

    # 결제 버튼

    cv2.rectangle(
        screen,
        (140, 250),
        (500, 330),
        (0, 180, 80),
        -1
    )

    cv2.putText(
        screen,
        "PRESS ENTER TO PAY",
        (190, 300),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        screen,
        "R : CANCEL",
        (250, 390),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (80, 80, 80),
        1
    )

    cv2.putText(
        screen,
        "Q : QUIT",
        (255, 430),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (80, 80, 80),
        1
    )

    return screen


# =========================================================
# 14. PAYMENT COMPLETE
# =========================================================

def drawPaymentComplete(frame):

    screen = np.zeros_like(frame)

    screen[:] = (
        245,
        245,
        245
    )

    # Check circle

    cv2.circle(
        screen,
        (320, 110),
        45,
        (0, 190, 90),
        -1
    )

    cv2.putText(
        screen,
        "OK",
        (295, 122),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # Payment complete

    cv2.putText(
        screen,
        "PAYMENT COMPLETE",
        (135, 210),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (30, 30, 30),
        2
    )

    cv2.putText(
        screen,
        f"PAID  W{paid_total:,}",
        (220, 260),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (70, 70, 70),
        2
    )

    cv2.putText(
        screen,
        "THANK YOU!",
        (230, 330),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 180, 80),
        2
    )

    cv2.putText(
        screen,
        "SPACE : NEW ORDER",
        (210, 410),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (80, 80, 80),
        1
    )

    return screen


# =========================================================
# 15. CAMERA
# =========================================================

cap = cv2.VideoCapture(0)

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    640
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    480
)

cap.set(
    cv2.CAP_PROP_BUFFERSIZE,
    1
)


# =========================================================
# 16. WINDOW
# =========================================================

cv2.namedWindow(
    "Smart Checkout",
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    "Smart Checkout",
    960,
    720
)


# =========================================================
# 17. MAIN LOOP
# =========================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera error")
        break

    # 좌우 반전
    frame = cv2.flip(
        frame,
        1
    )

    # =====================================================
    # PAYMENT COMPLETE
    # =====================================================

    if payment_complete:

        screen = drawPaymentComplete(
            frame
        )

        cv2.imshow(
            "Smart Checkout",
            screen
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            break

        elif key == 32:

            resetCart()

            payment_complete = False
            payment_mode = False
            paid_total = 0

        continue


    # =====================================================
    # PAYMENT MODE
    # =====================================================

    if payment_mode:

        screen = drawPaymentScreen(
            frame
        )

        cv2.imshow(
            "Smart Checkout",
            screen
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            break

        elif key == ord("r"):

            resetCart()

            payment_mode = False

        elif key == 13:

            paid_total = getTotal()

            payment_complete = True

            payment_mode = False

        continue


    # =====================================================
    # DETECTION
    # =====================================================

    try:

        output, ratio, pad_x, pad_y = processImage(
            frame
        )

        current_product, detected_box, confidence = getBestProduct(
            output
        )

    except Exception as e:

        print("Detection error:", e)

        current_product = None
        detected_box = None
        confidence = 0


    # =====================================================
    # DRAW BOUNDING BOX
    # =====================================================

    if current_product is not None:

        drawBoundingBox(
            frame,
            detected_box,
            current_product,
            confidence,
            ratio,
            pad_x,
            pad_y
        )


    # =====================================================
    # AUTO ADD TO CART
    # =====================================================

    if current_product is not None:

        # 새로운 상품이 보이면 타이머 시작

        if candidate_product != current_product:

            candidate_product = current_product

            candidate_start_time = time.time()

        # 같은 상품이 일정 시간 유지됨

        elif (
            time.time() - candidate_start_time
            >= STABLE_TIME
        ):

            # 아직 추가하지 않은 경우

            if last_added_product != current_product:

                cart.append(
                    current_product
                )

                last_added_product = current_product

                # 알림 메시지

                cart_message = (
                    "ADDED TO CART : "
                    + product_name.get(
                        current_product,
                        "Unknown"
                    )
                )

                cart_message_time = time.time()

    else:

        # 물체가 사라지면 다시 추가 가능

        candidate_product = None

        candidate_start_time = 0

        last_added_product = None


    # =====================================================
    # CURRENT PRODUCT DISPLAY
    # =====================================================

    if current_product is not None:

        name = product_name.get(
            current_product,
            "Unknown"
        )

        price = product_price.get(
            current_product,
            0
        )

        cv2.putText(
            frame,
            f"PRODUCT: {name}",
            (20, 440),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"PRICE: W{price:,}",
            (20, 470),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1
        )




    # =====================================================
    # ITEM ADDED MESSAGE
    # =====================================================

    drawCartMessage(
        frame
    )


    # =====================================================
    # CART PANEL
    # =====================================================

    display = drawCartPanel(
        frame
    )


    # =====================================================
    # SHOW
    # =====================================================

    cv2.imshow(
        "Smart Checkout",
        display
    )


    # =====================================================
    # KEY
    # =====================================================

    key = cv2.waitKey(1) & 0xFF

    # Q : 종료

    if key == ord("q"):

        break

    # R : 장바구니 초기화

    elif key == ord("r"):

        resetCart()

    # ENTER : 결제

    elif key == 13:

        if len(cart) > 0:

            payment_mode = True


# =========================================================
# 18. CLEAN UP
# =========================================================

cap.release()

cv2.destroyAllWindows()