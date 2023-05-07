#!python3
# author: Moises Garin (moises.garin@uvic.cat)
# created: 30th of February, 2021
# import os.path
# import re
# from numpy.core.records import _OrderedCounter
# from numpy.core.numeric import ones_like
# from numpy.ma import empty_like
import math
import time
import os
import numpy as np
from pylab import *
from scipy.cluster.vq import kmeans, kmeans2
from scipy.interpolate import splev, splrep, make_interp_spline, make_lsq_spline, UnivariateSpline
import cv2 as cv
from matplotlib import pyplot as plt
from B_contour_util import *  # Particular import
import C_augmentation as ag
import Excel_Data as ed  # Particular import

close('all')


def robust_bbox_search(im, pout=10):  # 1):
    """
    pout = percentaje of pixels left out of the boinding rect.
    """
    if pout <= 0:
        delta = 1e-3
    else:
        delta = pout / 400

    sx = np.sum(im, 0)  # sum of pixels in x
    sy = np.sum(im, 1)

    cx = cumsum(sx)
    cy = cumsum(sy)

    ##    fig = figure()
    ##    ax = fig.add_subplot(211)
    ##    ax.plot(sx)
    ##    ax = fig.add_subplot(212)
    ##    ax.plot(sy)

    ##    # Discard percetaje points.
    ##    xmin = nonzero(cx>delta*cx[-1])[0][0]
    ##    xmax = nonzero(cx>(1-delta)*cx[-1])[0][0]
    ##    ymin = nonzero(cy>delta*cy[-1])[0][0]
    ##    ymax = nonzero(cy>(1-delta)*cy[-1])[0][0]

    # This is a bit more clever. Tries to not discard
    # pixels from the main body. 
    i = np.nonzero(np.logical_and(cx < delta * cx[-1], sx != 0))[0]
    if i.size:
        xmin = i[-1]
    else:
        xmin = 0
    i = np.nonzero(np.logical_and(cx > (1 - delta) * cx[-1], sx != 0))[0]
    if i.size:
        xmax = i[0]
    else:
        xmax = sx.size - 1
    i = np.nonzero(np.logical_and(cy < delta * cy[-1], sy != 0))[0]
    if i.size:
        ymin = i[-1]
    else:
        ymin = 0
    i = np.nonzero(np.logical_and(cy > (1 - delta) * cy[-1], sy != 0))[0]
    if i.size:
        ymax = i[0]
    else:
        ymax = sy.size - 1

    w = xmax - xmin
    h = ymax - ymin

    return (xmin, ymin, w, h)


def get_init_centers(bbox, N=None, pos=None):
    """
    Return the estimated positions of the egg centers
    given a boudning box and the number of eggs.

    bbox = (x,y,widht,height) The bounding box of eggs.
    N = Number of eggs.
        6 - half a dozen.
        12 - a dozen,
        (nr,nc) - an arrange of nr rows by nc columns of eggs.
        None - Try to deduce if half a dozen or a dozen. 
    pos = {'h','v'} Position of the egg case.
        'h' -> horizontal.
        'v' -> vertical
        None -> Try to deduce.
    """
    x, y, w, h = bbox

    # Autodetect horientation.
    if pos is None:
        if h > w:
            pos = 'v'
        else:
            pos = 'h'

    # I want to do all calculations as if horizontal.
    if pos == 'v':
        x, y, w, h = y, x, h, w

    # Autodetect number of eggs. 
    if N is None:
        # Go for either half a dozen or a dozen.
        if w / h >= 2:
            N = 12
        else:
            N = 6

    # Get number of rows and columns. 
    if N == 12:
        nr, nc = 2, 6
    elif N == 6:
        nr, nc = 2, 3
    else:
        nr, nc = N

    # Notice that we calculate (x,y) coordinates.
    cc = [((c + .5) / nc * w + x, (r + .5) / nr * h + y) for r in range(nr) for c in range(nc)]

    # If image horientation was vertical,
    # transpose again the resulting coordinates.
    if pos == 'v':
        cc = [(y, x) for x, y in cc]

    return cc


def segment_eggs(im, ksize=0,
                 thmin=(1, 30, 90), thmax=(28, 255, 255),
                 pout=10,
                 rmin=0.65, rmax=1.25,
                 channel='s',
                 debug=True, case_params=False):
    """
    Detect eggs on the image.
    Return the instance segmentated image.
    
    PARAMETERS
    ----------
    im : ndarray
        Image to process.

    ksize : integer
        Size of the kernel for initial blurring.
        Use a 0 or negative value for no blurring.
    thmin, thmax : 3-uple
        lower and upper bounds (hsv) for binarizing the image.
    pout : float
        percentaje of pixels left out of the boinding rect.
    
    case_params : dict
        Parameters defining the egg-cases: number of eggs, horientation, etc...

    RETURNS:
    --------
    labels : ndarray
    """

    # Esta función combina kameans para determinar los
    # centros de N huevos y luego aplica watershed.
    # No tengo claro que aplicar watershed sea lo mejor
    # siempre. Por ejemplo, si la binarización inicial
    # es muy biena, el mismom resultado del kmeans
    # puede servir, seguido de detección de contornos
    # y algún mecanismo para rellenar posibles zonas
    # que no estuvieran bien binarizadas iniciamente. 

    # Some filtering,
    # probably can go along without it.
    if ksize > 0:
        im = cv.GaussianBlur(im, (ksize, ksize), 0)

    ##    b,g,r = cv.split(im)
    ##    r = cv.equalizeHist(r)
    ##    g = cv.equalizeHist(g)
    ##    b = cv.equalizeHist(b)
    ##    im = cv.merge((b,g,r))

    # Do all processing in hsv space.
    hsv = cv.cvtColor(im, cv.COLOR_BGR2HSV_FULL)  # COLOR_BGR2HSV)

    debug = True

    # Visualize the HSV channels.
    if debug == False:
        figure()
        ax = subplot(131)
        ax.imshow(hsv[:, :, 0], cmap=plt.get_cmap('hsv'))
        ax.set_title('H')
        ax = subplot(132, sharex=ax, sharey=ax)
        ax.imshow(hsv[:, :, 1])
        ax.set_title('S')
        ax = subplot(133, sharex=ax, sharey=ax)
        ax.imshow(hsv[:, :, 2])
        ax.set_title('V')

    # Binarize as a function of hue, but also setting a minimum
    # value for saturation and value. 
    im2 = cv.inRange(hsv, thmin, thmax)

    if debug == False:
        fig = figure()
        ax = fig.add_subplot(111)
        ax.imshow(im2)
        ax.set_title('binarized image')

    # Calculate a bounding rectangle of the detected region.
    # bbox = cv.boundingRect(im2)
    bbox = robust_bbox_search(im2)

    # Determine the estimated initial position of the egg
    # centers from the bbox. This works best in the farm's
    # images, as background and egg-case are almost colorless.
    # May be the position of the centers could be just 
    # defined a priori if the position of the cases is very
    # repeatible and if speed is an issue.
    start_centers = get_init_centers(bbox)
    start_centers = np.array(start_centers)

    # Visualize the bounding box.
    if debug == False:
        figure()
        img = im.copy()
        x, y, w, h = bbox
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 10)
        imshow(img)
    # Refine center positions using kmeans.
    ii = cv.findNonZero(im2)
    data = ii.reshape(-1, 2).astype(np.float32)

    # Calculate number of points to use in kmeans.
    # Limit to 500pts by egg.
    # This greatly reduces the execution time without
    # an appreciable loss in the precision of the resuts. 
    npt = min((data.shape[0], len(start_centers) * 500))
    factor = data.shape[0] / npt  # correction factor for area calculation.
    ii = choice(data.shape[0], npt, replace=False)  # replace=False avoid repetitions, but slower
    data = data[ii]
    centers, labels = kmeans2(data, start_centers, thresh=1, check_finite=False)

    if debug == False:
        print(f'Points after decimation: {ii.size}')
        print('List of egg centers:')
        print(centers)

    # Aproximate the egg radious using the median of the
    # area of the eggs.
    area = []
    for i in range(len(start_centers)):
        area.append(np.sum(labels == i))
    area.sort()
    area = area[int(len(area) / 2)]
    r = math.sqrt(area * factor / math.pi)
    if debug == False:
        print(f"estimated egg radius {r}")

    labels = np.ones_like(im[:, :, 0], dtype=np.int32)

    for i, center in enumerate(centers):
        # print("center:",center)
        center = (int(center[0]), int(center[1]))
        cv.circle(labels, tuple(center), int(r * rmax), 0, -1)
        cv.circle(labels, tuple(center), int(r * rmin), i + 2, -1)
        # fig = figure()
        # ax = fig.add_subplot(111)
        # ax.imshow(labels,cmap=get_cmap('jet'))
        # ax.set_title('Watershed seed regions')

    #
    # Using the saturation channel seems to be best
    # for egg delimitation using watershed, at least
    # with the images from the farm where the
    # centered and well focused.
    #
    if channel == 'h':
        img = cv.cvtColor(hsv[:, :, 0], cv.COLOR_GRAY2BGR)
    elif channel == 's':
        img = cv.cvtColor(hsv[:, :, 1], cv.COLOR_GRAY2BGR)
    elif channel == 'v':
        img = cv.cvtColor(hsv[:, :, 2], cv.COLOR_GRAY2BGR)
    elif channel == 'hsv':
        img = hsv
    elif channel == 'bin':
        img = cv.cvtColor(im2, cv.COLOR_GRAY2BGR)
    else:
        assert False

    # figure()
    # imshow(img)

    # figure()
    # imshow(labels)

    cv.watershed(img, labels)

    if debug == False:
        fig = figure()
        ax = fig.add_subplot(111)
        ax.imshow(labels, cmap=get_cmap('jet'))
        ax.set_title('Watershed segmentation result')

    if debug == False:
        print("showing")
        show()

    return labels, centers


def segmentation_view(im, labels, sf=1, lw=2, gs=0, s=10):
    """ Makes a pretty representation of the segmented image
    sf: Scale factor of the labels.
    lw: Line width of the contour. zero for not drawing.
    gs: gauss smoothing of the labels before countoring.
    s: contour smoothing factor. s>=1; s=1 -> No smoothing. 
    """

    imo = im.copy()
    aim = np.empty_like(labels, dtype=np.ubyte)  # (im.shape[:2],dtype=ubyte)
    cmap = get_cmap('hsv')

    contours = []
    for n in range(2, amax(labels) + 1):
        cv.compare(labels, n, cv.CMP_EQ, dst=aim)

        # Smooth the label before contouring.
        # Slow but improves a bit the contour.
        if gs:
            aim = cv.GaussianBlur(aim, (gs, gs), 0)
            cv.compare(aim, 128, cv.CMP_GE, dst=aim)

        c, _ = cv.findContours(aim, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)  # .CHAIN_APPROX_SIMPLE)

        if s > 1:
            c[0] = smooth_convex_spl(c[0], s=s)
        # c[0] = cv.approxPolyDP(c[0],2,True) #simplify contour
        contours.append(c)
    # Scale to real image resolution.
    if sf != 1:
        for cs in contours:
            for i in range(len(cs)):
                cs[i] = (cs[i] * sf).astype(int32)

    # Draw contours as transparent regions.
    for i, c in enumerate(contours):
        color = cmap(i / (len(contours) - 1))
        color = [c * 255 for c in color[2::-1]]
        cv.drawContours(im, c, -1, color=color, thickness=-1)
    im = cv.addWeighted(imo, .6, im, .4, 0)

    # if lw>0
    for i, c in enumerate(contours):
        color = cmap(i / (len(contours) - 1))
        color = [c * 255 for c in color[2::-1]]
        cv.drawContours(im, c, -1, color=color, thickness=lw)

    return im


def batch(path, sf=1, s=1, gs=3, ksize=1):
    image_folder = r'C:\Users\Erick\Desktop\TFG\Documentos\flash'
    ##########################Read Excel File ###############################################
    excel_folder = r'C:\Users\Erick\Desktop\TFG\Documentos\analisi_rafa.xlsx'
    dic = ed.read_excel(excel_folder)  # // Abrimos  un excel que contenga guardado todos los huevos malos
    ############################Obtein a photo of a single egg###############################
    count = 0  # // Iniciamos un contador
    for i in os.listdir(image_folder):
        pos = ed.bad_egg(i,
                         dic)  # Comprobamos si en la foto hay huevos malos, en caso afrimativo nos devuelve una lista con las posiciones de lo contrario retorna un False.
        # imo = cv.imread(os.path.join(image_folder,i))#// Guardamos la imagen
        imo = cv.imread(r'C:\Users\Erick\Desktop\TFG\Documentos\flash\IMG_20210309_100352.jpg')  # borrar !!!!!!!
        name = os.path.splitext(i)  # // Obtenemos el nombre de la imagen
        imo = cv.resize(imo, None, fx=.25, fy=.25)  # // Ajustamos la grandaria de la imagen
        labels, _ = segment_eggs(imo, ksize=ksize)  # // Etiquetamos cada huevo que hay en la imagen
        for e in range(2, np.amax(labels) + 1):
            labels, _ = segment_eggs(imo,
                                     ksize=ksize)  # // Volvemos a etquitar cada huevo que hay en la imagen dado que en cada iteracion esta informacion se pierde
            labels[labels != e] = 0;
            labels[labels == e] = 255  # // Creamos una mascara binaria para un huevo según el orden etiquetado
            labels = np.uint8(
                labels)  # // Para poder aplicar ciertas operaciones debemos convertirlo al formato adecuado.
            B, _, _ = cv.split(imo)  # // Separamos la imagen en sus diferentes canales de colores
            B = B & labels  # // Aplicamos la mascara sobre un canal de color, este procesamiento previo solo se utiliza para poder aplicar la funcion robust_rotation.
            x, y, z, n = robust_bbox_search(B,
                                            pout=10)  # // Sobre el huevo aplicamos la funcion que nos crea cuadrado con el huevo dentro
            if x > 20 and y > 20:  # Nos aseguramos que el huevo este dentro del cuadrodado, dado que en ocasiones por la posicion del huevo dentro de la imagen puede dar error.
                img_2 = ag.shift_rotation(x, y, z, n, imo)  # // Aplicamos diversos efectos
                cv.imshow("Image", img_2)
                cv.waitKey(0)
                black = ag.black_pixels(img_2)  # Verificamos que no haya pixeles negros en la imagen.
                if not black:
                    if pos:  # // Comprobamos que la actual imagen contenga huevos malos
                        if e - 1 in pos:  # // Si la imagen contiene  huevos malos la funcion "bad_eggs" nos devueleve una lista con la posicion de los huevos malos. Verificamos que la posicion del huevo malo este dentro de la lista
                            directory = r'C:\Users\Erick\Desktop\TFG\Samples'  # Malos
                            os.chdir(directory)  # // Guardamos la imagen del huevo en el directorio especificado.
                            cv.imwrite("Malo" + '_{}.jpg'.format(count), img_2)
                        else:  # // Good Egg
                            directory = r'C:\Users\Erick\Desktop\TFG\Samples'  # Buenos
                            os.chdir(directory)  # // Guardamos la imagen del huevo en el directorio especificado.
                            cv.imwrite("Bueno" + '_{}.jpg'.format(count), img_2)
                    else:  # // The currently image there aren't bad eggs
                        directory = r'C:\Users\Erick\Desktop\TFG\Samples'  # Buenos
                        # directory = r'C:\Users\Erick\Desktop\TFG\Huevos\Samples'#Buenos
                        os.chdir(directory)  # // Guardamos la imagen del huevo en el directorio especificado.
                        cv.imwrite("Bueno" + '_{}.jpg'.format(count),
                                   img_2)  # // Antes de guardarlo le damos un nombre y un formato
                else:
                    pass
            else:
                pass
            count += 1
    ##########################################################################################
    print("[+] Data Complete........................................")


def testing_contour():
    directory = r'C:\Users\alexb\OneDrive - Universitat de Vic\Escritorio\moises_nov_2021\IMG_20211121_141248450.jpg'
    im = cv.imread(directory, cv.IMREAD_COLOR)
    print("Nivel 1............")
    im = cv.resize(im, None, fx=.2, fy=.2, interpolation=cv.INTER_AREA)
    print("Nivel 2............")
    labels = segment_eggs(im, ksize=0, debug=False)
    print("Nivel 3............")
    aim = np.empty_like(labels, dtype=np.ubyte)
    print("Nivel 4............")
    cv.compare(labels, 2, cv.CMP_EQ, dst=aim)
    print("Nivel 5............")
    figure()
    imshow(labels)
    show()
    out = cv.findContours(aim, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    c, _ = out
    cc = c[0]
    c = c[0]
    c = c.reshape(-1, 2)
    x = c[:, 0]
    y = c[:, 1]
    dx = np.diff(x)
    dy = np.diff(y)
    dp = np.sqrt(dx * dx + dy * dy)
    p = np.empty_like(x)
    p[0] = 0
    p[1:] = cumsum(dp)
    s = 10000
    spx = splrep(p, x, s=s)
    spy = splrep(p, y, s=s)

    figure()
    plot(x, y)
    plot(splev(p, spx), splev(p, spy))

    figure()
    # plot(p,x)
    plot(p, splev(p, spx))
    # plot(p,y)
    plot(p, splev(p, spy))

    show()


def test_image():
    im = cv.imread('egg_sample.jpg')

    tick = time.time()
    ##    width = int(im.shape[1]/8)
    ##    height = int(im.shape[0]/8)
    ##    size = (width,height)
    im = cv.resize(im, None, fx=.5, fy=.5, interpolation=cv.INTER_AREA)
    labels, _ = segment_eggs(im, ksize=0, debug=True)
    tock = time.time()
    print(tock - tick)
    # figure()
    # imshow(labels)
    sim = segmentation_view(im, labels)
    cv.imshow('result', sim)
    cv.waitKey(0)
    cv.destroyAllWindows()
    show()


def test_smooth():
    im = cv.imread("../fotos/IMG_20210309_105424.jpg", cv.IMREAD_COLOR)
    labels = segment_eggs(im, ksize=5, debug=False)

    aim = np.empty_like(labels, dtype=ubyte)
    cv.compare(labels, 2, cv.CMP_EQ, dst=aim)
    _, c, _ = cv.findContours(aim, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    c = c[0]
    c2 = smooth_convex_spl(c)
    # c2 = smooth_contour_spl(c)
    x = c[:, 0, 0]
    y = c[:, 0, 1]
    x2 = c2[:, 0, 0]
    y2 = c2[:, 0, 1]
    figure()
    plot(x, y)
    plot(x2, y2)
    show()


if __name__ == '__main__':
    batch(path=None, sf=3, s=50, gs=1)  # s=50
    # testing_contour()
    # test_smooth()
    # test_image()
    # segment_eggs(img)
