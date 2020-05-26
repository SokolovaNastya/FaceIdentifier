import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import scipy.ndimage
import imageio
import imutils
import math
from sklearn.cluster import KMeans
import bisect
import scipy.spatial
import random
import progressbar
import pandas as pd


def grayscale(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def dodge(front,back):
    result = front*255/(255-back)
    result[result>255] = 255
    result[back==255] = 255
    return result.astype('uint8')

def limit_size(img, max_x, max_y=0):
    if max_x == 0:
        return img

    if max_y == 0:
        max_y = max_x

    ratio = min(1.0, float(max_x) / img.shape[1], float(max_y) / img.shape[0])

    if ratio != 1.0:
        shape = (int(img.shape[1] * ratio), int(img.shape[0] * ratio))
        return cv2.resize(img, shape, interpolation=cv2.INTER_AREA)
    else:
        return img


def clipped_addition(img, x, _max=255, _min=0):
    if x > 0:
        mask = img > (_max - x)
        img += x
        np.putmask(img, mask, _max)
    if x < 0:
        mask = img < (_min - x)
        img += x
        np.putmask(img, mask, _min)


def regulate(img, hue=0, saturation=0, luminosity=0):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
    if hue < 0:
        hue = 255 + hue
    hsv[:, :, 0] += hue
    clipped_addition(hsv[:, :, 1], saturation)
    clipped_addition(hsv[:, :, 2], luminosity)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR_FULL)

def compute_color_probabilities(pixels, palette, k=9):
    distances = scipy.spatial.distance.cdist(pixels, palette.colors)
    maxima = np.amax(distances, axis=1)

    distances = maxima[:, None] - distances
    summ = np.sum(distances, 1)
    distances /= summ[:, None]

    distances = np.exp(k*len(palette)*distances)
    summ = np.sum(distances, 1)
    distances /= summ[:, None]

    return np.cumsum(distances, axis=1, dtype=np.float32)


def color_select(probabilities, palette):
    r = random.uniform(0, 1)
    i = bisect.bisect_left(probabilities, r)
    return palette[i] if i < len(palette) else palette[-1]


def randomized_grid(h, w, scale):
    assert (scale > 0)

    r = scale//2

    grid = []
    for i in range(0, h, scale):
        for j in range(0, w, scale):
            y = random.randint(-r, r) + i
            x = random.randint(-r, r) + j

            grid.append((y % h, x % w))

    random.shuffle(grid)
    return grid

class ColorPalette:
    def __init__(self, colors, base_len=0):
        self.colors = colors
        self.base_len = base_len if base_len > 0 else len(colors)

    @staticmethod
    def from_image(img, n, max_img_size=200, n_init=10):
        # scale down the image to speedup kmeans
        img = limit_size(img, max_img_size)

        clt = KMeans(n_clusters=n, n_jobs=1, n_init=n_init)
        clt.fit(img.reshape(-1, 3))

        return ColorPalette(clt.cluster_centers_)

    def extend(self, extensions):
        extension = [regulate(self.colors.reshape((1, len(self.colors), 3)).astype(np.uint8), *x).reshape((-1, 3)) for x
                     in
                     extensions]

        return ColorPalette(np.vstack([self.colors.reshape((-1, 3))] + extension), self.base_len)

    def to_image(self):
        cols = self.base_len
        rows = int(math.ceil(len(self.colors) / cols))

        res = np.zeros((rows * 80, cols * 80, 3), dtype=np.uint8)
        for y in range(rows):
            for x in range(cols):
                if y * cols + x < len(self.colors):
                    color = [int(c) for c in self.colors[y * cols + x]]
                    cv2.rectangle(res, (x * 80, y * 80), (x * 80 + 80, y * 80 + 80), color, -1)

        return res

    def __len__(self):
        return len(self.colors)

    def __getitem__(self, item):
        return self.colors[item]


class VectorField:
    def __init__(self, fieldx, fieldy):
        self.fieldx = fieldx
        self.fieldy = fieldy

    @staticmethod
    def from_gradient(gray):
        fieldx = cv2.Scharr(gray, cv2.CV_32F, 1, 0) / 15.36
        fieldy = cv2.Scharr(gray, cv2.CV_32F, 0, 1) / 15.36

        return VectorField(fieldx, fieldy)

    def get_magnitude_image(self):
        res = np.sqrt(self.fieldx ** 2 + self.fieldy ** 2)

        return (res * 255 / np.max(res)).astype(np.uint8)

    def smooth(self, radius, iterations=1):
        s = 2 * radius + 1
        for _ in range(iterations):
            self.fieldx = cv2.GaussianBlur(self.fieldx, (s, s), 0)
            self.fieldy = cv2.GaussianBlur(self.fieldy, (s, s), 0)

    def direction(self, i, j):
        return math.atan2(self.fieldy[i, j], self.fieldx[i, j])

    def magnitude(self, i, j):
        return math.hypot(self.fieldx[i, j], self.fieldy[i, j])


if __name__ == '__main__':

    folder_path = '/home/datasets/images/MS1M/raw/'
    save_path = '/home/student/asokolova/vggface2_ms1m_best_worst_gen/generated_data/'

    pic_arr = np.load('/home/student/asokolova/vggface2_ms1m_best_worst_gen/vggface2_ms1m_Good_path.npz')['x']

    count = 0
    cur_name = ''
    tmp = 0
    image_ids = []
    class_res = []

    for img_name_init in pic_arr:
        img_name_init = img_name_init.astype(str)
        print(img_name_init)
        path = folder_path + img_name_init
        img_name = img_name_init.replace('/','-')
        img_name_spl = img_name.split('-')[0]

        if tmp == 0:
            tmp = 100
            cur_name = img_name_spl
            
        if count % 9 == 0:
            gray = Image.open(path).convert('LA')
            out_img = save_path + img_name + '_gray.png'
            gray.save(out_img)
            image_ids.append(out_img)
            class_res.append(0)

        if count % 9 == 1:
            img = imageio.imread(path)
            gray = lambda rgb: np.dot(rgb[..., :3], [0.21 , 0.72, 0.07])
            gray = gray(img)
            inverted_img = 255 - gray
            blur_img = scipy.ndimage.filters.gaussian_filter(inverted_img, sigma=5)
            final_img = dodge(blur_img, gray)
            out_img = save_path + img_name + '_painted_pencil.png'
            plt.imsave(out_img, final_img, cmap ='gray', vmin = 0, vmax = 255)
            image_ids.append(out_img)
            class_res.append(0)

        if count % 9 == 2:
            bgr_img = cv2.imread(path)
            img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            blur = cv2.medianBlur(img,7)
            out_img = save_path + img_name + '_blur.png'
            plt.imsave(out_img, blur, vmin=0, vmax=255)
            image_ids.append(out_img)
            class_res.append(0)

        if count % 9 == 3:
            img_n = cv2.imread(path)[...,::-1]/255.0
            noise = np.random.normal(loc=0, scale=1, size=img_n.shape)
            noisy = np.clip((img_n + noise*0.5),0,1)
            out_img = save_path + img_name + '_noisy.png'
            plt.imsave(out_img, noisy, vmin=0, vmax=255)
            image_ids.append(out_img)
            class_res.append(0)

        if count % 9 == 4:
            img2 = img_n*2
            n2 = np.clip(np.where(img2 <= 1, (img2*(1 + noise*0.2)), (1-img2+1)*(1 + noise*0.2)*-1 + 2)/2, 0,1)
            out_img = save_path + img_name + '_noisy2.png'
            plt.imsave(out_img, n2, vmin=0, vmax=255)
            image_ids.append(out_img)
            class_res.append(0)

        if count % 9 == 5:
            rotated_90 = imutils.rotate_bound(img, 90)
            out_img = save_path + img_name + '_rotated90.png'
            plt.imsave(out_img, rotated_90, vmin=0, vmax=255)
            image_ids.append(out_img)
            class_res.append(0)

        #rotated_180 = imutils.rotate_bound(img, 180)
        #out_img = save_path + 'rotated180/' + img_name + '_rotated180.png'
        #plt.imsave(out_img, rotated_180, vmin=0, vmax=255)

        if count % 9 == 6:
            rotated_270 = imutils.rotate_bound(img, 270)
            out_img = save_path + img_name + '_rotated270.png'
            plt.imsave(out_img, rotated_270, vmin=0, vmax=255)
            image_ids.append(out_img)
            class_res.append(0)

        if count % 9 == 7:
            w = img.shape[0]
            h = img.shape[1]
            crop_img = img[int(w/3) : int(w*2/3), int(h/3) : int(h*2/3)]
            out_img = save_path + img_name + '_crop.png'
            plt.imsave(out_img, crop_img, vmin=0, vmax=255)
            image_ids.append(out_img)
            class_res.append(0)

        if count % 9 == 8:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            palette_size = 10
            palette = ColorPalette.from_image(img, palette_size)
            palette = palette.extend([(0, 50, 0), (15, 30, 0), (-15, 30, 0)])
            gradient = VectorField.from_gradient(gray)
            gradient_smoothing_radius = int(round(max(img.shape) / 50))
            gradient.smooth(gradient_smoothing_radius)
            res = cv2.medianBlur(img, 11)
            grid = randomized_grid(img.shape[0], img.shape[1], scale=3)
            batch_size = 500
            bar = progressbar.ProgressBar()
            stroke_scale = int(math.ceil(max(img.shape) / 1000))
            for h in bar(range(0, len(grid), batch_size)):
                pixels = np.array([img[x[0], x[1]] for x in grid[h:min(h + batch_size, len(grid))]])
                color_probabilities = compute_color_probabilities(pixels, palette, k=9)
                for i, (y, x) in enumerate(grid[h:min(h + batch_size, len(grid))]):
                    color = color_select(color_probabilities[i], palette)
                    angle = math.degrees(gradient.direction(y, x)) + 90
                    length = int(round(stroke_scale + stroke_scale * math.sqrt(gradient.magnitude(y, x))))
                    cv2.ellipse(res, (x, y), (length, stroke_scale), angle, 0, 360, color, -1, cv2.LINE_AA)
            out_img = save_path + img_name + '_painted_color.png'
            plt.imsave(out_img, res, vmin=0, vmax=255)
            image_ids.append(out_img)
            class_res.append(0)
        
        if cur_name != img_name_spl:
            count = count + 1
            cur_name = img_name_spl
            
    data = {'id': image_ids, 'Good_or_bad': class_res}
    results = pd.DataFrame(data)
    results.to_csv("/home/student/asokolova/vggface2_ms1m_best_worst_gen/classes_vggface2_ms1m_Gen_class.csv", index=False)