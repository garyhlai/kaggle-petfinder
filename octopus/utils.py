from matplotlib import pyplot as plt

def show_data_img(data):
    plt.figure()
    plt.imshow(data['image'])
    plt.figtext(0, 0, f"score: {data['target']}", fontsize = 10)