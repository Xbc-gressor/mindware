from mindware.components.feature_engineering.transformations.base_transformer import *


class Ordinal_encoder(Transformer):
    type = 'dky'

    def __init__(self):
        super().__init__('ordinal_encoder')
        self.input_type = CATEGORICAL
        self.output_type = CATEGORICAL
        self.compound_mode = 'replace'
        

    @ease_trans
    def operate(self, input_datanode: DataNode, target_fields=None):
        from sklearn.preprocessing import OrdinalEncoder
    
        if target_fields is None:
            target_fields = collect_fields(input_datanode.feature_types, self.input_type)


        X, y = input_datanode.data

        self.target_fields = target_fields
        X_input = X[:,target_fields]

        if self.model is None:
            self.model = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            self.model.fit(X_input)
        _X  = self.model.transform(X_input)

        return _X