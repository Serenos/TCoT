import cv2

# 打开摄像头，参数0表示默认的摄像头设备。
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

while True:
    # 从摄像头读取一帧
    ret, frame = cap.read()
    
    # 如果正确读取到帧，ret为True
    if not ret:
        print("无法接收帧（可能已达到视频末尾？）")
        break
    
    # 显示结果帧
    cv2.imshow('Camera Feed', frame)
    
    # 按'q'键退出循环
    if cv2.waitKey(1) == ord('q'):
        break

# 完成后释放VideoCapture对象
cap.release()
cv2.destroyAllWindows()