Python 3.11.5 (tags/v3.11.5:cce6ba9, Aug 24 2023, 14:38:34) [MSC v.1936 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
import numpy as np
import pandas as pd
%pylab inline
SyntaxError: invalid syntax
import matplotlib.pyplot as plt
plt.show()
import plotly.graph_objects as go
Traceback (most recent call last):
  File "<pyshell#5>", line 1, in <module>
    import plotly.graph_objects as go
ModuleNotFoundError: No module named 'plotly'
import plotly.graph_objects as go
Traceback (most recent call last):
  File "D:\py\python\Lib\importlib\metadata\__init__.py", line 563, in from_name
    return next(cls.discover(name=name))
StopIteration

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<pyshell#6>", line 1, in <module>
    import plotly.graph_objects as go
  File "D:\py\python\Lib\site-packages\plotly\__init__.py", line 34, in <module>
    __version__ = importlib.metadata.version("plotly")
  File "D:\py\python\Lib\importlib\metadata\__init__.py", line 1008, in version
    return distribution(distribution_name).version
  File "D:\py\python\Lib\importlib\metadata\__init__.py", line 981, in distribution
    return Distribution.from_name(distribution_name)
  File "D:\py\python\Lib\importlib\metadata\__init__.py", line 565, in from_name
    raise PackageNotFoundError(name)
importlib.metadata.PackageNotFoundError: No package metadata was found for plotly
from sklearn.cluster import KMeans
img = plt.imread('using_kmeans_for_color_compression_tulips_photo.jpeg')
Traceback (most recent call last):
  File "<pyshell#8>", line 1, in <module>
    img = plt.imread('using_kmeans_for_color_compression_tulips_photo.jpeg')
  File "D:\py\python\Lib\site-packages\matplotlib\pyplot.py", line 2607, in imread
    return matplotlib.image.imread(fname, format)
  File "D:\py\python\Lib\site-packages\matplotlib\image.py", line 1512, in imread
    with img_open(fname) as image:
  File "D:\py\python\Lib\site-packages\PIL\Image.py", line 3465, in open
    fp = builtins.open(filename, "rb")
FileNotFoundError: [Errno 2] No such file or directory: 'using_kmeans_for_color_compression_tulips_photo.jpeg'
print(img.shape)
Traceback (most recent call last):
  File "<pyshell#9>", line 1, in <module>
    print(img.shape)
NameError: name 'img' is not defined
NameError: name 'img' is not defined
SyntaxError: invalid syntax
img = plt.imread('using_kmeans_for_color_compression_tulips_photo.jpeg')
Traceback (most recent call last):
  File "<pyshell#11>", line 1, in <module>
    img = plt.imread('using_kmeans_for_color_compression_tulips_photo.jpeg')
  File "D:\py\python\Lib\site-packages\matplotlib\pyplot.py", line 2607, in imread
    return matplotlib.image.imread(fname, format)
  File "D:\py\python\Lib\site-packages\matplotlib\image.py", line 1512, in imread
    with img_open(fname) as image:
  File "D:\py\python\Lib\site-packages\PIL\Image.py", line 3465, in open
    fp = builtins.open(filename, "rb")
FileNotFoundError: [Errno 2] No such file or directory: 'using_kmeans_for_color_compression_tulips_photo.jpeg'
import matplotlib.pyplot as plt
img = plt.imread(r"C:\Users\Administrator\Desktop\using_kmeans_for_color_compression_tulips_photo.jpeg")
print(img.shape)
(320, 240, 3)
plt.imshow(img)
<matplotlib.image.AxesImage object at 0x00000214FA3852D0>
plt.axis('off');
(np.float64(-0.5), np.float64(239.5), np.float64(319.5), np.float64(-0.5))
plt.imshow(img)
<matplotlib.image.AxesImage object at 0x00000214FA36ED10>
plt.axis('off')
(np.float64(-0.5), np.float64(239.5), np.float64(319.5), np.float64(-0.5))
plt.show()
img_flat = img.reshape(img.shape[0]*img.shape[1], 3)
img_flat[:5, :]
array([[211, 197,  38],
       [199, 181,  21],
       [178, 154,   0],
       [185, 152,   0],
       [184, 145,   0]], dtype=uint8)
img_flat.shape
(76800, 3)
img_flat_df = pd.DataFrame(img_flat, columns = ['r', 'g', 'b'])
img_flat_df.head()
     r    g   b
0  211  197  38
1  199  181  21
2  178  154   0
3  185  152   0
4  184  145   0
trace = go.Scatter3d(x = img_flat_df.r,
                     y = img_flat_df.g,
                     z = img_flat_df.b,
                     mode='markers',
                     marker=dict(size=1,
                                 color=['rgb({},{},{})'.format(r,g,b) for r,g,b
                                        in zip(img_flat_df.r.values,
                                               img_flat_df.g.values,
                                               img_flat_df.b.values)],
                                 opacity=0.5))
Traceback (most recent call last):
  File "<pyshell#35>", line 1, in <module>
    trace = go.Scatter3d(x = img_flat_df.r,
NameError: name 'go' is not defined
def plot_color_3d_scatter(img_flat_df):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(img_flat_df.r,
                         img_flat_df.g,
                         img_flat_df.b,
                         c=[f'#{r:02x}{g:02x}{b:02x}' for r, g, b in
                           zip(img_flat_df.r.values,
                               img_flat_df.g.values,
                               img_flat_df.b.values)],
                         alpha=0.5,
                         s=1)
    ax.set_xlabel('R')
    ax.set_ylabel('G')
    ax.set_zlabel('B')
    ax.set_title('3D Color Space Visualization')
    plt.tight_layout()
    plt.show()
def extract_colors_from_image(image_path, sample_size=1000):
    
SyntaxError: invalid syntax
def extract_colors_from_image(image_path, sample_size=1000):
    """
    Extract colors from an image for 3D visualization
    img = Image.open(image_path)
    img = img.convert('RGB')
    pixels = img_array.reshape(-1, 3)
    if len(pixels) > sample_size:
       indices = np.random.choice(len(pixels), sample_size, replace=False)
       pixels = pixels[indices]
    color_df = pd.DataFrame(pixels, columns=['r', 'g', 'b'])
    return color_df
def plot_color_3d_scatter(img_flat_df):
    """
    Create a 3D scatter plot of colors
    
SyntaxError: invalid decimal literal
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
colors = [f'#{int(r):02x}{int(g):02x}{int(b):02x}' for r, g, b in
          zip(img_flat_df.r, img_flat_df.g, img_flat_df.b)]
scatter = ax.scatter(img_flat_df.r,
                     img_flat_df.g,
                     img_flat_df.b,
                     c=colors,
                     alpha=0.7,
                     s=20)
ax.set_xlabel('Red')
Text(0.5, 0, 'Red')
ax.set_ylabel('Green')
Text(0.5, 0.5, 'Green')
ax.set_zlabel('Blue')
Text(0.5, 0, 'Blue')
ax.set_title('3D Color Space Visualization of Tulip Image')
Text(0.5, 0.92, '3D Color Space Visualization of Tulip Image')
ax.set_xlim(0, 255)
(0.0, 255.0)
ax.set_ylim(0, 255)
(0.0, 255.0)
ax.set_zlim(0, 255)
(0.0, 255.0)
ax.view_init(20, 45)
plt.tight_layout()
plt.show()
tulip_colors = extract_colors_from_image('tulips.jpg', sample_size=2000)
plot_color_3d_scatter(tulip_colors)
plt.show()
kmeans = KMeans(n_clusters=1, random_state=42).fit(img_flat)
img_flat1 = img_flat.copy()
for i in np.unique(kmeans.labels_):
    img_flat1[kmeans.labels_==i,:] = kmeans.cluster_centers_[i]
img1 = img_flat1.reshape(img.shape)
SyntaxError: invalid syntax
img1 = img_flat1.reshape(img.shape)
plt.imshow(img1)
<matplotlib.image.AxesImage object at 0x0000021485568F90>
plt.axis('off');
(np.float64(-0.5), np.float64(239.5), np.float64(319.5), np.float64(-0.5))
column_means = img_flat.mean(axis=0)

column_means = img_flat.mean(axis=0)
print('column means: ', column_means)
column means:  [125.60802083  78.90632813  43.45473958]
print('cluster centers: ', kmeans.cluster_centers_)
cluster centers:  [[125.60802083  78.90632813  43.45473958]]
trace = go.Scatter3d(x = img_flat_df.r,
                     y = img_flat_df.g,
                     z = img_flat_df.b,
                     mode='markers',
                     marker=dict(size=1,
                                 color=['rgb({},{},{})'.format(r,g,b) for
...                                         r,g,b in zip(img_flat_df.r.values,
...                                                      img_flat_df.g.values,
...                                                      img_flat_df.b.values)],
...                                  opacity=0.5))
Traceback (most recent call last):
  File "<pyshell#110>", line 1, in <module>
    trace = go.Scatter3d(x = img_flat_df.r,
NameError: name 'go' is not defined
>>> trace = go.Scatter3d(
...     x=img_flat_df.r,
...     y=img_flat_df.g,
...     z=img_flat_df.b,
...     mode='markers',
...     marker=dict(
...         size=1,
...         color=['rgb({},{},{})'.format(r,g,b) for r,g,b in
...                zip(img_flat_df.r.values,
...                    img_flat_df.g.values,
...                    img_flat_df.b.values)],
...         opacity=0.5
...     )
... )
...                          
Traceback (most recent call last):
  File "<pyshell#124>", line 1, in <module>
    trace = go.Scatter3d(
NameError: name 'go' is not defined
>>> layout = go.Layout(
...     margin=dict(l=0, r=0, b=0, t=0),
...     scene=dict(
...         xaxis_title='R',
...         yaxis_title='G',
...         zaxis_title='B'
...     )
... )
Traceback (most recent call last):
  File "<pyshell#132>", line 1, in <module>
    layout = go.Layout(
NameError: name 'go' is not defined
