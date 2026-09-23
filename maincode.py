import cv2
import numpy as np
import time
import ai_edge_litert.interpreter as tflite


# =========================================================
# 1. YOLO 모델 설정
# =========================================================

modelPath = "best.tflite"

# TFLite 모델 불러오기
interpreter = tflite.Interpreter(model_path=modelPath)

# 모델 실행에 필요한 메모리 할당
interpreter.allocate_tensors()

# 입력 / 출력 텐서 정보 가져오기
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]["shape"]

print("Input shape :", input_shape)
print("Output shape:", output_details[0]["shape"])
print("Input dtype :", input_details[0]["dtype"])
print("Output dtype:", output_details[0]["dtype"])


# =========================================================
# 2. 상품 정보
# =========================================================

# YOLO 클래스 번호와 상품 이름 연결
product_name = {
    0: "Coffee",
    1: "Water",
    2: "Letsbe",
    3: "Cola",
    4: "Pocachip",
    5: "Yukgaejang",
    6: "Khancho",
    7: "Pepero"
}

# YOLO 클래스 번호와 상품 가격 연결
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
# 3. YOLO 설정
# =========================================================

IMG_SIZE = 320

# 상품 인식 최소 신뢰도
CONF_TH = 0.4

# 상품을 몇 초 동안 유지해야 장바구니에 추가할지
STABLE_TIME = 2.0


# =========================================================
# 4. 프로그램 상태 변수
# =========================================================

# 장바구니
cart = []

# 현재 인식 중인 상품
candidate_product = None
candidate_start_time = 0

# 마지막으로 추가한 상품
last_added_product = None

# 장바구니 메시지
cart_message = ""
cart_message_product = None
cart_message_time = 0

# 메시지 표시 시간
CART_MESSAGE_DURATION = 1.5

# 결제 화면 상태
payment_mode = False

# 결제 완료 상태
payment_complete = False

# 결제 완료 금액
paid_total = 0

# 결제 완료 후 영수증에 사용할 장바구니
paid_cart = []


# =========================================================
# 5. Letterbox
# =========================================================

def letterbox(image, new_shape=(320, 320)):

    # 원본 이미지 크기
    shape = image.shape[:2]

    # 원본 비율을 유지하면서 크기 계산
    ratio = min(
        new_shape[0] / shape[0],
        new_shape[1] / shape[1]
    )

    new_unpad = (
        int(round(shape[1] * ratio)),
        int(round(shape[0] * ratio))
    )

    # 추가할 여백 크기
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]

    dw /= 2
    dh /= 2

    # 이미지 크기 조정
    resized = cv2.resize(
        image,
        new_unpad,
        interpolation=cv2.INTER_LINEAR
    )

    # 여백 크기 계산
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    # 회색 여백 추가
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
# 6. 이미지 처리 및 YOLO 추론
# =========================================================

def processImage(frame):

    # 320 x 320으로 변환
    image, ratio, pad_x, pad_y = letterbox(
        frame,
        (IMG_SIZE, IMG_SIZE)
    )

    # BGR → RGB
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # 0~1 사이 값으로 정규화
    image = image.astype(
        np.float32
    ) / 255.0

    # 모델 입력 형태 확인
    expected_shape = input_details[0]["shape"]

    # NCHW 형식
    if len(expected_shape) == 4 and expected_shape[1] == 3:

        image = np.transpose(
            image,
            (2, 0, 1)
        )

        image = np.expand_dims(
            image,
            axis=0
        )

    # NHWC 형식
    elif len(expected_shape) == 4 and expected_shape[3] == 3:

        image = np.expand_dims(
            image,
            axis=0
        )

    else:

        raise ValueError(
            f"Unsupported input shape: {expected_shape}"
        )

    # 입력 데이터 타입 확인
    input_dtype = input_details[0]["dtype"]

    if input_dtype == np.float32:

        input_data = image.astype(
            np.float32
        )

    elif input_dtype == np.uint8:

        input_data = (
            image * 255
        ).astype(np.uint8)

    elif input_dtype == np.int8:

        input_data = (
            image * 127
        ).astype(np.int8)

    else:

        input_data = image.astype(
            input_dtype
        )

    # 모델에 이미지 입력
    interpreter.set_tensor(
        input_details[0]["index"],
        input_data
    )

    # YOLO 실행
    interpreter.invoke()

    # 결과 가져오기
    output = interpreter.get_tensor(
        output_details[0]["index"]
    )

    return output, ratio, pad_x, pad_y


# =========================================================
# 7. 가장 높은 확률의 상품 찾기
# =========================================================

def getBestProduct(output):

    # 차원 정리
    data = np.squeeze(output)

    # 2차원 데이터가 아니면 실패
    if data.ndim != 2:
        return None, None, 0

    # 데이터 방향 맞추기
    if data.shape[0] < data.shape[1]:
        data = data.T

    best_class = None
    best_box = None
    best_conf = 0

    # 각각의 검출 결과 확인
    for row in data:

        if len(row) < 5:
            continue

        # Bounding Box
        x = row[0]
        y = row[1]
        w = row[2]
        h = row[3]

        # 클래스별 점수
        class_scores = row[4:]

        # 가장 높은 클래스 선택
        class_id = int(
            np.argmax(class_scores)
        )

        # 해당 클래스의 신뢰도
        confidence = float(
            class_scores[class_id]
        )

        # 현재까지 가장 높은 신뢰도인지 확인
        if confidence > best_conf:

            best_conf = confidence

            best_class = class_id

            best_box = (
                float(x),
                float(y),
                float(w),
                float(h)
            )

    # 신뢰도가 기준보다 낮으면 인식하지 않음
    if best_conf < CONF_TH:

        return None, None, 0

    return best_class, best_box, best_conf


# =========================================================
# 8. 총 가격 계산
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
# 9. 장바구니 초기화
# =========================================================

def resetCart():

    global cart
    global candidate_product
    global candidate_start_time
    global last_added_product
    global cart_message
    global cart_message_product
    global cart_message_time

    cart = []

    candidate_product = None
    candidate_start_time = 0

    last_added_product = None

    cart_message = ""
    cart_message_product = None
    cart_message_time = 0


# =========================================================
# 10. Bounding Box 표시
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

    if detected_box is None:
        return

    if current_product is None:
        return

    x, y, w, h = detected_box

    # 원본 영상 좌표로 변환
    box_x1 = int(
        (x - w / 2 - pad_x) / ratio
    )

    box_y1 = int(
        (y - h / 2 - pad_y) / ratio
    )

    box_x2 = int(
        (x + w / 2 - pad_x) / ratio
    )

    box_y2 = int(
        (y + h / 2 - pad_y) / ratio
    )

    # 화면 범위를 벗어나지 않도록 제한
    box_x1 = max(
        0,
        min(
            frame.shape[1] - 1,
            box_x1
        )
    )

    box_y1 = max(
        0,
        min(
            frame.shape[0] - 1,
            box_y1
        )
    )

    box_x2 = max(
        0,
        min(
            frame.shape[1] - 1,
            box_x2
        )
    )

    box_y2 = max(
        0,
        min(
            frame.shape[0] - 1,
            box_y2
        )
    )

    # Bounding Box
    cv2.rectangle(
        frame,
        (box_x1, box_y1),
        (box_x2, box_y2),
        (255, 255, 255),
        2
    )

    # 상품 이름
    name = product_name.get(
        current_product,
        "Unknown"
    )

    font = cv2.FONT_HERSHEY_SIMPLEX

    font_scale = 0.65
    thickness = 2

    label = (
        f"{name}  "
        f"{confidence * 100:.1f}%"
    )

    (text_w, text_h), _ = cv2.getTextSize(
        label,
        font,
        font_scale,
        thickness
    )

    label_x = box_x1

    if box_y1 > text_h + 15:

        label_y = box_y1

    else:

        label_y = box_y1 + text_h + 15

    # 라벨 배경
    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (
            label_x,
            label_y - text_h - 14
        ),
        (
            label_x + text_w + 20,
            label_y + 5
        ),
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

    # 상품 이름
    cv2.putText(
        frame,
        name,
        (
            label_x + 8,
            label_y - 5
        ),
        font,
        font_scale,
        (255, 255, 255),
        thickness
    )

    # 신뢰도
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
# 11. 상품 추가 / 삭제 메시지
# =========================================================

def drawCartMessage(frame):

    # 메시지 표시 시간이 지나면 종료
    if (
        time.time() -
        cart_message_time
        >= CART_MESSAGE_DURATION
    ):
        return

    frame_h, frame_w = frame.shape[:2]

    box_w = 420
    box_h = 130

    x1 = (
        frame_w -
        box_w
    ) // 2

    y1 = (
        frame_h -
        box_h
    ) // 2

    x2 = x1 + box_w
    y2 = y1 + box_h

    # 어두운 배경
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

    font = cv2.FONT_HERSHEY_SIMPLEX

    # 제목
    title_scale = 0.85
    title_thickness = 2

    (title_w, title_h), _ = cv2.getTextSize(
        cart_message,
        font,
        title_scale,
        title_thickness
    )

    title_x = (
        x1 +
        (box_w - title_w) // 2
    )

    title_y = y1 + 50

    cv2.putText(
        frame,
        cart_message,
        (title_x, title_y),
        font,
        title_scale,
        (255, 255, 255),
        title_thickness
    )

    # 상품 정보
    if cart_message_product is not None:

        name = product_name.get(
            cart_message_product,
            "Unknown"
        )

        price = product_price.get(
            cart_message_product,
            0
        )

        product_text = (
            f"{name} - W{price:,}"
        )

        product_scale = 0.65
        product_thickness = 2

        (product_w, product_h), _ = cv2.getTextSize(
            product_text,
            font,
            product_scale,
            product_thickness
        )

        product_x = (
            x1 +
            (box_w - product_w) // 2
        )

        product_y = y1 + 95

        cv2.putText(
            frame,
            product_text,
            (
                product_x,
                product_y
            ),
            font,
            product_scale,
            (0, 220, 255),
            product_thickness
        )


# =========================================================
# 12. 상품 인식 진행률
# =========================================================

def drawRecognitionProgress(frame):

    # 인식 중인 상품이 없으면 표시하지 않음
    if candidate_product is None:
        return

    # 인식된 시간
    elapsed = (
        time.time() -
        candidate_start_time
    )

    # 진행률
    progress = (
        elapsed /
        STABLE_TIME
    )

    # 0~1 범위로 제한
    progress = max(
        0.0,
        min(1.0, progress)
    )

    # 진행률 바 위치
    bar_x = 20
    bar_y = 385

    bar_width = 300
    bar_height = 20

    # 진행률 배경
    cv2.rectangle(
        frame,
        (
            bar_x,
            bar_y
        ),
        (
            bar_x + bar_width,
            bar_y + bar_height
        ),
        (80, 80, 80),
        -1
    )

    # 현재 진행된 부분
    progress_width = int(
        bar_width * progress
    )

    cv2.rectangle(
        frame,
        (
            bar_x,
            bar_y
        ),
        (
            bar_x + progress_width,
            bar_y + bar_height
        ),
        (0, 220, 100),
        -1
    )

    # 상품 이름
    name = product_name.get(
        candidate_product,
        "Unknown"
    )

    cv2.putText(
        frame,
        f"Recognizing: {name}",
        (20, 350),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # 퍼센트
    percent = int(
        progress * 100
    )

    cv2.putText(
        frame,
        f"{percent}%",
        (330, 402),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


# =========================================================
# 13. 상품 인식 실패 안내
# =========================================================

def drawDetectionGuide(
    frame,
    current_product
):

    # 상품이 인식되고 있으면 표시하지 않음
    if current_product is not None:
        return

    box_x1 = 15
    box_y1 = 335
    box_x2 = 430
    box_y2 = 385

    # 반투명 배경
    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (
            box_x1,
            box_y1
        ),
        (
            box_x2,
            box_y2
        ),
        (30, 30, 30),
        -1
    )

    cv2.addWeighted(
        overlay,
        0.8,
        frame,
        0.2,
        0,
        frame
    )

    # 안내 제목
    cv2.putText(
        frame,
        "PRODUCT NOT DETECTED",
        (30, 357),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    # 안내 내용
    cv2.putText(
        frame,
        "Show the product to the camera",
        (30, 378),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 220, 255),
        1
    )


# =========================================================
# 14. 장바구니 패널
# =========================================================

def drawCartPanel(frame):

    panel_width = 320

    # 오른쪽 장바구니 패널
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

    # 제목
    cv2.putText(
        panel,
        "SHOPPING CART",
        (25, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # 전체 상품 개수
    cv2.putText(
        panel,
        f"ITEMS : {len(cart)}",
        (200, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 220, 255),
        1
    )

    # 구분선
    cv2.line(
        panel,
        (20, 55),
        (300, 55),
        (100, 100, 100),
        1
    )

    y = 90

    # 같은 상품끼리 개수 계산
    counts = {}

    for product in cart:

        if product not in counts:
            counts[product] = 0

        counts[product] += 1

    # 상품 출력
    for product, count in counts.items():

        name = product_name.get(
            product,
            "Unknown"
        )

        price = product_price.get(
            product,
            0
        )

        item_total = (
            price *
            count
        )

        # 상품명
        cv2.putText(
            panel,
            f"{name} x{count}",
            (25, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (230, 230, 230),
            1
        )

        # 가격
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

    # 구분선
    cv2.line(
        panel,
        (20, y + 5),
        (300, y + 5),
        (100, 100, 100),
        1
    )

    # 총 금액
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

    # 조작 방법
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
        (25, 410),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 180, 180),
        1
    )

    cv2.putText(
        panel,
        "D : DELETE LAST ITEM",
        (25, 430),
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

    # 카메라 화면 + 장바구니
    result = np.hstack(
        (
            frame,
            panel
        )
    )

    return result


# =========================================================
# 15. 결제 화면
# =========================================================

def drawPaymentScreen(frame):

    screen = np.zeros_like(frame)

    # 배경
    screen[:] = (
        245,
        245,
        245
    )

    # 제목
    cv2.putText(
        screen,
        "PAYMENT",
        (210, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (30, 30, 30),
        3
    )

    # 총 금액
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

    # 뒤로가기
    cv2.putText(
        screen,
        "BACKSPACE : ADD MORE ITEMS",
        (160, 390),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (80, 80, 80),
        1
    )

    # 종료
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
# 16. 결제 완료 + 영수증 화면
# =========================================================

def drawPaymentComplete(frame):

    screen = np.zeros_like(frame)

    # 배경
    screen[:] = (
        245,
        245,
        245
    )

    font = cv2.FONT_HERSHEY_SIMPLEX

    # -----------------------------------------------------
    # 제목
    # -----------------------------------------------------

    cv2.putText(
        screen,
        "PAYMENT COMPLETE",
        (155, 45),
        font,
        0.8,
        (30, 30, 30),
        2
    )

    # -----------------------------------------------------
    # 영수증 영역
    # -----------------------------------------------------

    receipt_x1 = 100
    receipt_y1 = 65

    receipt_x2 = 540
    receipt_y2 = 385

    cv2.rectangle(
        screen,
        (
            receipt_x1,
            receipt_y1
        ),
        (
            receipt_x2,
            receipt_y2
        ),
        (255, 255, 255),
        -1
    )

    cv2.rectangle(
        screen,
        (
            receipt_x1,
            receipt_y1
        ),
        (
            receipt_x2,
            receipt_y2
        ),
        (180, 180, 180),
        2
    )

    # -----------------------------------------------------
    # 영수증 제목
    # -----------------------------------------------------

    cv2.putText(
        screen,
        "SMART CHECKOUT",
        (205, 95),
        font,
        0.5,
        (50, 50, 50),
        2
    )

    # 구분선
    cv2.line(
        screen,
        (125, 110),
        (515, 110),
        (180, 180, 180),
        1
    )

    # -----------------------------------------------------
    # 상품 개수 계산
    # -----------------------------------------------------

    counts = {}

    for product in paid_cart:

        if product not in counts:
            counts[product] = 0

        counts[product] += 1

    # -----------------------------------------------------
    # 상품 출력
    # -----------------------------------------------------

    y = 140

    for product, count in counts.items():

        name = product_name.get(
            product,
            "Unknown"
        )

        price = product_price.get(
            product,
            0
        )

        item_total = (
            price *
            count
        )

        # 상품 이름
        cv2.putText(
            screen,
            f"{name} x{count}",
            (130, y),
            font,
            0.48,
            (50, 50, 50),
            1
        )

        # 상품 가격
        cv2.putText(
            screen,
            f"W{item_total:,}",
            (405, y),
            font,
            0.48,
            (50, 50, 50),
            1
        )

        y += 27

    # -----------------------------------------------------
    # 총 금액 영역
    # -----------------------------------------------------

    cv2.line(
        screen,
        (125, y + 3),
        (515, y + 3),
        (180, 180, 180),
        1
    )

    # 상품 개수
    cv2.putText(
        screen,
        f"ITEMS : {len(paid_cart)}",
        (130, y + 32),
        font,
        0.45,
        (80, 80, 80),
        1
    )

    # TOTAL
    cv2.putText(
        screen,
        "TOTAL",
        (130, y + 62),
        font,
        0.58,
        (30, 30, 30),
        2
    )

    # 총 결제 금액
    cv2.putText(
        screen,
        f"W{paid_total:,}",
        (390, y + 62),
        font,
        0.58,
        (0, 170, 80),
        2
    )

    # -----------------------------------------------------
    # THANK YOU
    # 화면 아래쪽에 맞게 위치만 조정
    # -----------------------------------------------------

    cv2.putText(
        screen,
        "THANK YOU!",
        (235, 415),
        font,
        0.65,
        (0, 170, 80),
        2
    )

    # -----------------------------------------------------
    # 새 주문
    # 화면 아래쪽에 맞게 위치만 조정
    # -----------------------------------------------------

    cv2.putText(
        screen,
        "SPACE : NEW ORDER",
        (210, 448),
        font,
        0.48,
        (80, 80, 80),
        1
    )

    # -----------------------------------------------------
    # 종료
    # 화면 아래쪽에 맞게 위치만 조정
    # -----------------------------------------------------

    cv2.putText(
        screen,
        "Q : QUIT",
        (270, 470),
        font,
        0.43,
        (100, 100, 100),
        1
    )

    return screen


# =========================================================
# 17. 카메라 설정
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

# 카메라 버퍼 최소화
cap.set(
    cv2.CAP_PROP_BUFFERSIZE,
    1
)


# =========================================================
# 18. OpenCV 창 설정
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
# 19. 메인 프로그램
# =========================================================

while True:

    # 카메라 프레임 가져오기
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
    # 결제 완료 화면
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

        # Q → 종료
        if key == ord("q"):

            break

        # SPACE → 새 주문
        elif key == 32:

            resetCart()

            payment_complete = False
            payment_mode = False

            paid_total = 0
            paid_cart = []

        continue


    # =====================================================
    # 결제 화면
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

        # Q → 종료
        if key == ord("q"):

            break

        # -------------------------------------------------
        # Backspace → 상품 추가 화면으로 돌아가기
        # -------------------------------------------------

        elif key == 8:

            payment_mode = False

            # 상품을 다시 인식할 수 있도록 초기화
            candidate_product = None
            candidate_start_time = 0
            last_added_product = None

        # -------------------------------------------------
        # Enter → 결제
        # -------------------------------------------------

        elif key == 13:

            # 현재 장바구니를 영수증용으로 저장
            paid_cart = cart.copy()

            # 총 결제 금액 저장
            paid_total = getTotal()

            # 결제 완료 화면으로 이동
            payment_complete = True
            payment_mode = False

        continue


    # =====================================================
    # YOLO 상품 인식
    # =====================================================

    try:

        output, ratio, pad_x, pad_y = processImage(
            frame
        )

        current_product, detected_box, confidence = getBestProduct(
            output
        )

    except Exception as e:

        print(
            "Detection error:",
            e
        )

        current_product = None
        detected_box = None
        confidence = 0


    # =====================================================
    # Bounding Box 표시
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
    # 상품 인식 및 장바구니 추가
    # =====================================================

    if current_product is not None:

        # 새로운 상품이 인식된 경우
        if candidate_product != current_product:

            candidate_product = current_product

            candidate_start_time = time.time()

        # 같은 상품이 계속 인식되고 있는 경우
        elif (
            time.time() -
            candidate_start_time
            >= STABLE_TIME
        ):

            # 아직 추가하지 않은 상품이면 장바구니에 추가
            if last_added_product != current_product:

                cart.append(
                    current_product
                )

                # 마지막 추가 상품 저장
                last_added_product = current_product

                # 추가 메시지
                cart_message = "ITEM ADDED"

                cart_message_product = current_product

                cart_message_time = time.time()


    # =====================================================
    # 상품이 인식되지 않은 경우
    # =====================================================

    else:

        candidate_product = None

        candidate_start_time = 0

        last_added_product = None


    # =====================================================
    # 현재 인식 상품 정보
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

        # 상품 이름
        cv2.putText(
            frame,
            f"PRODUCT: {name}",
            (20, 440),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        # 상품 가격
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
    # 상품 인식 실패 안내
    # =====================================================

    drawDetectionGuide(
        frame,
        current_product
    )


    # =====================================================
    # 상품 인식 진행률
    # =====================================================

    drawRecognitionProgress(
        frame
    )


    # =====================================================
    # 상품 추가 / 삭제 메시지
    # =====================================================

    drawCartMessage(
        frame
    )


    # =====================================================
    # 장바구니 패널
    # =====================================================

    display = drawCartPanel(
        frame
    )


    # =====================================================
    # 화면 출력
    # =====================================================

    cv2.imshow(
        "Smart Checkout",
        display
    )


    # 키 입력
    key = cv2.waitKey(1) & 0xFF


    # =====================================================
    # Q → 종료
    # =====================================================

    if key == ord("q"):

        break


    # =====================================================
    # R → 장바구니 전체 초기화
    # =====================================================

    elif key == ord("r"):

        resetCart()


    # =====================================================
    # Enter → 결제 화면
    # =====================================================

    elif key == 13:

        # 상품이 하나 이상 있을 때만 결제
        if len(cart) > 0:

            payment_mode = True


    # =====================================================
    # D → 마지막 상품 삭제
    # =====================================================

    elif key == ord("d"):

        if len(cart) > 0:

            # 삭제할 상품 저장
            removed_product = cart.pop()

            # 다시 인식할 수 있도록 초기화
            last_added_product = None
            candidate_product = None
            candidate_start_time = 0

            # 삭제 메시지
            cart_message = "ITEM REMOVED"

            cart_message_product = removed_product

            cart_message_time = time.time()
        


# =========================================================
# 20. 종료
# =========================================================

cap.release()

cv2.destroyAllWindows()