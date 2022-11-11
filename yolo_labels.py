import cv2
import os
import numpy as np

# Draw annotations
class drawLabels():

    bboxColor = (0, 255, 0)

    def __init__(self, srcDir, desDir):
        self.src = srcDir
        self.des = desDir

    # Read file content and save to an array
    def extractContent(self, labelFile):
        if labelFile.endswith(".txt"):
            with open(labelFile, mode="r") as f:
                content = f.readlines()
                dataArr = np.zeros(shape=(len(content), 5)) # to store label file content
            # read each line and map it to numpy array row
            for idx, line in enumerate(content):
                data = np.array(list(map(float, line.split())))
                dataArr[idx] = data # concatenate rows

        return dataArr

    # Draw bounding box using coordinates from dataArr
    def drawBox(self, img, dataArr):
        for dataRow in dataArr:
            # get true bounding box sizes
            # since they are normalized in yolo labels
            x = int(dataRow[1] * img.shape[1])
            y = int(dataRow[2] * img.shape[0])
            w = int(dataRow[3] * img.shape[1])
            h = int(dataRow[4] * img.shape[0])

            topLeft = (x - w//2, y - h//2)
            bottomRight = (x + w//2, y + h//2)

            img = cv2.rectangle(img, topLeft, bottomRight, self.bboxColor, 2)
        
        return img

    # Main function
    # Draw bounding box for images from their labels given
    def visualize(self):
        imgs = os.listdir(os.path.join(self.src, "images"))
        annos = os.listdir(os.path.join(self.src, "labels"))
        imgs.sort()
        annos.sort()

        for idx, file in enumerate(imgs):
            img = cv2.imread(os.path.join(self.src, "images", file))
            if img is None:
                print(f"{file} is corrupted")
                break
            else:
                dataArr = self.extractContent(os.path.join(self.src, "labels", annos[idx]))
                output = self.drawBox(img, dataArr)
                cv2.imwrite(os.path.join(self.des, f"{idx}.jpg"), output)


if __name__ == "__main__":
    draw = drawLabels("/home/quocanh/Downloads/motorbike", "/home/quocanh/Downloads/motorbike/output")
    draw.visualize()