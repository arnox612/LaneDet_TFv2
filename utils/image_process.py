


class LaneDataset(Dataset):

    def __init__(self, csv_file, transform=None):
        super(LaneDataset, self).__init__()
        self.data = pd.read_csv(os.path.join(os.getcwd(), "data_list", csv_file), header=None,
                                  names=["image","label"])
        self.images = self.data["image"].values[1:]
        self.labels = self.data["label"].values[1:]

        self.transform = transform