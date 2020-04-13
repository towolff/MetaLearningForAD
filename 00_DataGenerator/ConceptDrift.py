import numpy as np
import pandas as pd
import GridUtils as gu


class Drifter:
    
    def __init__(self, grid, concept_drifts, load_mapping, unscaled_data, timestamp_list, agg_data):
        self.grid = grid
        self.initial_load_mapping = load_mapping
        self.unscaled_data = unscaled_data
        self.data = agg_data
        self.timestamp_list = timestamp_list
        self.events_manipulate_switch = concept_drifts['manipulate_switch']
        self.events_load_mapping = concept_drifts['load_mapping']
        self.events_cos_phi = concept_drifts['change_cos_phi']
        self.concept_drifts_all = concept_drifts
        self.events = []
        self.current_ts = 0
        self.last_ts = 0
        self.simulation_done = False

    def check_timestamps(self, ts):
        """
        returns three tuples for every concept drift type. Every tuple contains a boolean on position 0,
        determining wheter or not there are events for this timestamp.
        On position 1 the there is a list with events for this timestamp.
        """
        self.last_ts = self.current_ts
        self.current_ts = ts

        result_cd1, events_cd1 = self._check_manipulate_switch_ts(ts) 

        return result_cd1, events_cd1

    def manipulate_switch(self, event):
        """
        Manipulate Switches by given events. Meaning: Opening and closing switches
        :param event: dict, one event with Switch Id, Action and time idx.
        :return: returns the manipulated pandapower Grid.
        """
        self.events.append(event)
        switch_no = event['switch_id']
        switch_status = event['set_closed']
        ts = event['at_time_idx']

        print('+++' * 10)
        print('In step {} do:'.format(ts))
        print('Manipulate Switch #: {}'.format(switch_no))
        print('Set Switch to Status: {}'.format(switch_status))
        print('+++' * 10)
        self.grid.switch.at[switch_no, 'closed'] = switch_status
        return self.grid

    def manipulate_load_mapping(self):
        """
        Manipulates the load mapping. Loops over all given load mapping events and manipulates the data by
        multiplying new load mapping times unscaled data and finally replaces the old data by new manipulated data.
        :return: the manipulated data with new load mapping as pandas DataFrame
        """

        if self.last_ts > 0:
            return None

        for event in self.events_load_mapping:
            start_ts = self.timestamp_list[event['at_time_idx']]
            end_ts = self.timestamp_list[event['until_time_idx']]
            start_ts_i = event['at_time_idx']
            end_ts_i = event['until_time_idx']

            if end_ts_i == -1:
                ts_list = self.timestamp_list[start_ts_i:]
            else:
                end_ts_i += 1
                ts_list = self.timestamp_list[start_ts_i:end_ts_i]

            bus_id = event['bus_id']
            bus_tag = 'AGG_BUS_{}'.format(bus_id)
            new_load_mapping = event['load_mapping']
            old_load_mapping = self.initial_load_mapping[bus_id]

            print('+++' * 10)
            print('On {} change load mapping at bus: {} until {}'.format(start_ts, bus_id, end_ts))
            print('Old load mapping was: {}'.format(old_load_mapping))
            print('New load mapping is: {}'.format(new_load_mapping))

            assert len(new_load_mapping) == len(old_load_mapping)

            tmp_data = self.unscaled_data.loc[start_ts:end_ts]
            agg_data_p = np.dot(new_load_mapping, tmp_data.T)
            print('Manipulate {} samples!'.format(len(agg_data_p)))
            print('+++' * 10)

            q_list = []
            for val in agg_data_p:
                q = gu.compute_q(val)
                q_list.append(q)

            assert len(ts_list) == len(q_list) == len(agg_data_p)

            for t_step, q_val, p_val in zip(ts_list, q_list, agg_data_p):
                self.data.loc[(t_step,  bus_id-1)] = [bus_tag, p_val, q_val]

        return self.data

    def manipulate_cos_phi(self):
        if self.last_ts > 0:
            return None

        for e in self.events_cos_phi:
            interval, interval_i, ts_list, bus_id, bus_tag, load_mapping, load_ids, cos_phi = self._get_cos_phi_vars(e)

            print('+++' * 10)
            string = 'From {} (idx: {}) change cos phi from load {} at bus {} with cos_phi of {} ' \
                     'until {} (idx: {})'.format(interval[0], interval_i[0], load_ids, bus_id,
                                                 cos_phi, interval[1], interval_i[1])
            print(string)
            print('Use load mapping: {}'.format(load_mapping))

            tmp_data = self.unscaled_data.loc[interval[0]:interval[1]]
            print('Manipulate {} samples!'.format(len(tmp_data)))

            list_q = []
            list_p = []

            for i, val in enumerate(load_mapping):
                prod_p = np.dot(val, self.unscaled_data[self.unscaled_data.columns[i]].loc[interval[0]:interval[1]].T)
                list_p.append(prod_p)

                if i in load_ids:
                    reactive_power = gu.compute_q(prod_p, cos_phi=cos_phi)
                else:
                    reactive_power = gu.compute_q(prod_p)

                list_q.append(reactive_power)

            assert len(list_p) == len(list_q)

            summed_p = np.zeros(len(list_p[0]))
            summed_q = np.zeros(len(list_q[0]))

            for val_p, val_q in zip(list_p, list_q):
                summed_p += val_p
                summed_q += val_q

            assert len(summed_p) == len(summed_q) == len(tmp_data)

            ### replace old data by drifted data with q vals with new cos phi
            for t_step, q_val, p_val in zip(ts_list, summed_q, summed_p):
                self.data.loc[(t_step, bus_id - 1)] = [bus_tag, p_val, q_val]

            print('+++' * 10)

        return self.data

    def annotate_drifting_labels(self):
        label_l = np.zeros(len(self.timestamp_list))
        
        for event_type in self.concept_drifts_all.keys():
            for event in self.concept_drifts_all[event_type]:
                if event_type == "manipulate_switch" and event["set_closed"]:
                    start_idx = event["at_time_idx"]
                    end_idx = event['until_time_idx']
                    label_l[start_idx:end_idx] = 1.0
                elif event_type == "load_mapping" or event_type == "change_cos_phi":
                    start_idx = event["at_time_idx"]
                    end_idx = event["until_time_idx"]
                    
                    if event_type == "load_mapping":
                        label_l[start_idx:end_idx] = 2.0
                    elif event_type == "change_cos_phi":
                        label_l[start_idx:end_idx] = 3.0
        
        
        return label_l
    
    def _get_cos_phi_vars(self, event):
        start_ts = self.timestamp_list[event['at_time_idx']]
        end_ts = self.timestamp_list[event['until_time_idx']]
        start_ts_i = event['at_time_idx']
        end_ts_i = event['until_time_idx']

        if end_ts_i == -1:
            ts_list = self.timestamp_list[start_ts_i:]
        else:
            end_ts_i += 1
            ts_list = self.timestamp_list[start_ts_i:end_ts_i]

        bus_id = event['bus_id']
        bus_tag = 'AGG_BUS_{}'.format(bus_id)

        if event['load_mapping'] is None:
            load_mapping = self.initial_load_mapping[bus_id]
        else:
            load_mapping = event['load_mapping']

        load_id = event['load_id_in_load_mapping']
        cos_phi = event['cos_phi']

        return (start_ts, end_ts), (start_ts_i, end_ts_i), ts_list, bus_id, bus_tag, load_mapping, load_id, cos_phi

    def _check_manipulate_switch_ts(self, ts):
        """
        Check if in this ts an action is neccessary!
        :param ts: the current Timestamp as integer.
        :return: result as bool which is True if there is an event. And the actual event as dict.
        """
        events = []
        result = False
        for event in self.events_manipulate_switch:
            if event['at_time_idx'] == ts:
                events.append(event)
                result = True

        return result, events
