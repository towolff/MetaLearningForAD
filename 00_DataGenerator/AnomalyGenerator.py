import numpy as np


class AnomalyGenerator:
    def __init__(self, data, anomalies):
        self.data = data
        self.anomalies = anomalies
        
    def _create_single_anomaly(self, value, pos_neg, factor):
        val = factor * pos_neg + value
        return val
        
    def _get_pos_neg(self):
        np.random.seed()
        if np.random.randint(0, 100+1) >= 50:
            pos_neg = 1
        else:
            pos_neg = -1
        return pos_neg
    
    def _create_collective_anomaly(self, column, start_ts, end_ts, factor):
        selected_vals = self.data[column].iloc[start_ts:end_ts].copy()
        anomalous_vals = []
        label_list = [2] * len(selected_vals)
        if selected_vals is not None:
            pos_neg = self._get_pos_neg()
            for val in selected_vals:
                aval = self._create_single_anomaly(val, pos_neg, factor)
                anomalous_vals.append(aval)

            self.data[column].iloc[start_ts:end_ts] = anomalous_vals
            self.data['anomaly_labels'].iloc[start_ts:end_ts] = label_list
            
    def _create_point_anomaly(self, column, point_ts, factor):
        current_val = self.data[column].iloc[point_ts]
        pos_neg = self._get_pos_neg()
        factor = self.data[column].iloc[point_ts-1] / 100 * factor
        anomal_val = self._create_single_anomaly(current_val, pos_neg, factor)
        self.data[column].iloc[point_ts] = anomal_val
        self.data['anomaly_labels'].iloc[point_ts] = 1
    
    def _create_noise_anomaly(self, column, start_ts, end_ts, factor):
        selected_vals = self.data[column].iloc[start_ts:end_ts].copy()
        noise = np.random.normal(1,factor, len(selected_vals))
        anomalous_vals = selected_vals + noise
        self.data[column].iloc[start_ts:end_ts] = anomalous_vals
        self.data['anomaly_labels'].iloc[start_ts:end_ts] = [3] * len(selected_vals)
    
    def _collective_anomaly(self, collectivs):
        for feature in collectivs:
            col = feature['feature']
            timestamps = feature['timestamps']
            factors = feature['factors']
            assert len(timestamps) == len(factors)
            for event, fact in zip(timestamps, factors):
                start_ts = event[0]
                end_ts = event[1]
                factor = self.data[col].iloc[start_ts-1] / 100 * fact
                if end_ts > start_ts:
                    self._create_collective_anomaly(col, start_ts, end_ts, factor)
    
    def _point_anomaly(self, points):
        for feature in points:
            col = feature['feature']
            timestamps = feature['timestamps']
            factors = feature['factors']
            assert len(timestamps) == len(factors)
            for event, fact in zip(timestamps, factors):
                point_ts = event
                factor = self.data[col].iloc[point_ts-1] / 100 * fact
                self._create_point_anomaly(col, point_ts, factor)
    
    def _noise_anomaly(self, noises):
        for feature in noises:
            col = feature['feature']
            timestamps = feature['timestamps']
            factors = feature['factors']
            assert len(timestamps) == len(factors)
            for event, fact in zip(timestamps, factors):
                start_ts = event[0]
                end_ts = event[1]
                self._create_noise_anomaly(col, start_ts, end_ts, fact)

    def make_anomalous(self):
        self._collective_anomaly(self.anomalies['collective'])
        self._point_anomaly(self.anomalies['point'])
        self._noise_anomaly(self.anomalies['noise'])

        return self.data

    def get_anomalous_data(self):
        return self.data

    def validate_drifted_anomalie_idx(self):
        idx = 0
        for d, val in self.data.iterrows():
            if val['anomaly_labels'] > 0 and val['drift_labels'] > 0:
                print('Anomaly and Concept Drift at IDX: {} ({})'.format(idx, d))
            
            idx += 1
