class House:
    def __init__(self, data):
        self.id = data['OBJECTID']
        self.status = "vacant"
        self.price = data['PUBVALUE']