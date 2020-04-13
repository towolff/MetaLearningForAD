import pandas as pd
import pandapower as pp


class Executor:

    def __init__(self, grid):
        self.grid = grid
        
        self.line_cols = ['line_1_1_loading', 'line_2_3_loading', 'line_3_4_loading', 'line_4_5_loading', 
                          'line_5_6_loading', 'line_6_7_loading', 'line_7_8_loading','line_8_9_loading',
                          'line_9_10_loading', 'line_10_11_loading', 'line_3_8_loading', 'line_12_13_loading',
                          'line_13_14_loading', 'line_6_7_loading', 'line_11_14_loading', 'line_14_8_loading']

        self.trafo_cols = ['trafo_0_loading', 'trafo_1_loading']

        self.bus_cols_tmplt = ['bus_0_{}', 'bus_1_{}', 'bus_2_{}', 'bus_3_{}', 'bus_4_{}',
                               'bus_5_{}', 'bus_6_{}',  'bus_7_{}', 'bus_8_{}', 'bus_9_{}',
                               'bus_10_{}', 'bus_11_{}', 'bus_12_{}', 'bus_13_{}', 'bus_14_{}']
        
        self.bus_cols_vm_pu = [x.format('vm_pu') for x in self.bus_cols_tmplt]
        
        self.bus_cols_va_degree = [x.format('va_degree') for x in self.bus_cols_tmplt]
        
        self.bus_cols_p_mw = [x.format('p_mw') for x in self.bus_cols_tmplt]
        
        self.bus_cols_q_mvar = [x.format('q_mvar') for x in self.bus_cols_tmplt]
        
        self.switch_cols = ['switch_0_status', 'switch_1_status', 'switch_2_status',
                            'switch_3_status', 'switch_4_status', 'switch_5_status', 
                            'switch_6_status', 'switch_7_status']
        
        self.columns = self.line_cols + self.trafo_cols + self.bus_cols_vm_pu + self.bus_cols_va_degree + self.bus_cols_p_mw + self.bus_cols_q_mvar + self.switch_cols

        self.dict_data = {
             'line_1_1_loading': [],
             'line_2_3_loading': [],
             'line_3_4_loading': [],
             'line_4_5_loading': [],
             'line_5_6_loading': [],
             'line_7_8_loading': [],
             'line_8_9_loading': [],
             'line_9_10_loading': [],
             'line_10_11_loading': [],
             'line_3_8_loading': [],
             'line_12_13_loading': [],
             'line_13_14_loading': [],
             'line_6_7_loading': [],
             'line_11_14_loading': [],
             'line_14_8_loading': [],
             'trafo_0_loading': [],
             'trafo_1_loading': [],
             'bus_0_vm_pu':[],
             'bus_1_vm_pu': [],
             'bus_2_vm_pu': [],
             'bus_3_vm_pu': [],
             'bus_4_vm_pu': [],
             'bus_5_vm_pu': [],
             'bus_6_vm_pu': [],
             'bus_7_vm_pu': [],
             'bus_8_vm_pu': [],
             'bus_9_vm_pu': [],
             'bus_10_vm_pu': [],
             'bus_11_vm_pu': [],
             'bus_12_vm_pu': [],
             'bus_13_vm_pu': [],
             'bus_14_vm_pu': [],
             'bus_0_va_degree': [],
             'bus_1_va_degree': [],
             'bus_2_va_degree': [],
             'bus_3_va_degree': [],
             'bus_4_va_degree': [],
             'bus_5_va_degree': [],
             'bus_6_va_degree': [],
             'bus_7_va_degree': [],
             'bus_8_va_degree': [],
             'bus_9_va_degree': [],
             'bus_10_va_degree': [],
             'bus_11_va_degree': [],
             'bus_12_va_degree': [],
             'bus_13_va_degree': [],
             'bus_14_va_degree': [],
             'bus_0_p_mw': [],
             'bus_1_p_mw': [],
             'bus_2_p_mw': [],
             'bus_3_p_mw': [],
             'bus_4_p_mw': [],
             'bus_5_p_mw': [],
             'bus_6_p_mw': [],
             'bus_7_p_mw': [],
             'bus_8_p_mw': [],
             'bus_9_p_mw': [],
             'bus_10_p_mw': [],
             'bus_11_p_mw': [],
             'bus_12_p_mw': [],
             'bus_13_p_mw': [],
             'bus_14_p_mw': [],
             'bus_0_q_mvar': [],
             'bus_1_q_mvar': [],
             'bus_2_q_mvar': [],
             'bus_3_q_mvar': [],
             'bus_4_q_mvar': [],
             'bus_5_q_mvar': [],
             'bus_6_q_mvar': [],
             'bus_7_q_mvar': [],
             'bus_8_q_mvar': [],
             'bus_9_q_mvar': [],
             'bus_10_q_mvar': [],
             'bus_11_q_mvar': [],
             'bus_12_q_mvar': [],
             'bus_13_q_mvar': [],
             'bus_14_q_mvar': [],
             'switch_0_status': [],
             'switch_1_status': [],
             'switch_2_status': [],
             'switch_3_status': [],
             'switch_4_status': [],
             'switch_5_status': [],
             'switch_6_status': [],
             'switch_7_status': []
            }

    def update_grid(self, grid):
        self.grid = grid

    def step(self, data):
        init = 'auto'
        algorithm = 'bfsw'

        # update load values
        self.grid.load.update(data)

        # run power flow
        pp.runpp(self.grid, algorithm=algorithm, init=init)

        # save data
        self._collect_data()

        # return the latest version of the powergrid
        return self.grid

    def _collect_data(self):
        self.dict_data['line_1_1_loading'].append(self.grid.res_line.loading_percent.loc[0])
        self.dict_data['line_2_3_loading'].append(self.grid.res_line.loading_percent.loc[1])
        self.dict_data['line_3_4_loading'].append(self.grid.res_line.loading_percent.loc[2])
        self.dict_data['line_4_5_loading'].append(self.grid.res_line.loading_percent.loc[3])
        self.dict_data['line_5_6_loading'].append(self.grid.res_line.loading_percent.loc[4])
        self.dict_data['line_7_8_loading'].append(self.grid.res_line.loading_percent.loc[5])
        self.dict_data['line_8_9_loading'].append(self.grid.res_line.loading_percent.loc[6])
        self.dict_data['line_9_10_loading'].append(self.grid.res_line.loading_percent.loc[7])
        self.dict_data['line_10_11_loading'].append(self.grid.res_line.loading_percent.loc[8])
        self.dict_data['line_3_8_loading'].append(self.grid.res_line.loading_percent.loc[9])
        self.dict_data['line_12_13_loading'].append(self.grid.res_line.loading_percent.loc[10])
        self.dict_data['line_13_14_loading'].append(self.grid.res_line.loading_percent.loc[11])
        self.dict_data['line_6_7_loading'].append(self.grid.res_line.loading_percent.loc[12])
        self.dict_data['line_11_14_loading'].append(self.grid.res_line.loading_percent.loc[13])
        self.dict_data['line_14_8_loading'].append(self.grid.res_line.loading_percent.loc[14])

        self.dict_data['trafo_0_loading'].append(self.grid.res_trafo.loading_percent[0])
        self.dict_data['trafo_1_loading'].append(self.grid.res_trafo.loading_percent[1])

        self.dict_data['bus_0_vm_pu'].append(self.grid.res_bus.vm_pu[0])
        self.dict_data['bus_1_vm_pu'].append(self.grid.res_bus.vm_pu[1])
        self.dict_data['bus_2_vm_pu'].append(self.grid.res_bus.vm_pu[2])
        self.dict_data['bus_3_vm_pu'].append(self.grid.res_bus.vm_pu[3])
        self.dict_data['bus_4_vm_pu'].append(self.grid.res_bus.vm_pu[4])
        self.dict_data['bus_5_vm_pu'].append(self.grid.res_bus.vm_pu[5])
        self.dict_data['bus_6_vm_pu'].append(self.grid.res_bus.vm_pu[6])
        self.dict_data['bus_7_vm_pu'].append(self.grid.res_bus.vm_pu[7])
        self.dict_data['bus_8_vm_pu'].append(self.grid.res_bus.vm_pu[8])
        self.dict_data['bus_9_vm_pu'].append(self.grid.res_bus.vm_pu[9])
        self.dict_data['bus_10_vm_pu'].append(self.grid.res_bus.vm_pu[10])
        self.dict_data['bus_11_vm_pu'].append(self.grid.res_bus.vm_pu[11])
        self.dict_data['bus_12_vm_pu'].append(self.grid.res_bus.vm_pu[12])
        self.dict_data['bus_13_vm_pu'].append(self.grid.res_bus.vm_pu[13])
        self.dict_data['bus_14_vm_pu'].append(self.grid.res_bus.vm_pu[14])

        self.dict_data['bus_0_va_degree'].append(self.grid.res_bus.va_degree[0])
        self.dict_data['bus_1_va_degree'].append(self.grid.res_bus.va_degree[1])
        self.dict_data['bus_2_va_degree'].append(self.grid.res_bus.va_degree[2])
        self.dict_data['bus_3_va_degree'].append(self.grid.res_bus.va_degree[3])
        self.dict_data['bus_4_va_degree'].append(self.grid.res_bus.va_degree[4])
        self.dict_data['bus_5_va_degree'].append(self.grid.res_bus.va_degree[5])
        self.dict_data['bus_6_va_degree'].append(self.grid.res_bus.va_degree[6])
        self.dict_data['bus_7_va_degree'].append(self.grid.res_bus.va_degree[7])
        self.dict_data['bus_8_va_degree'].append(self.grid.res_bus.va_degree[8])
        self.dict_data['bus_9_va_degree'].append(self.grid.res_bus.va_degree[9])
        self.dict_data['bus_10_va_degree'].append(self.grid.res_bus.va_degree[10])
        self.dict_data['bus_11_va_degree'].append(self.grid.res_bus.va_degree[11])
        self.dict_data['bus_12_va_degree'].append(self.grid.res_bus.va_degree[12])
        self.dict_data['bus_13_va_degree'].append(self.grid.res_bus.va_degree[13])
        self.dict_data['bus_14_va_degree'].append(self.grid.res_bus.va_degree[14])

        self.dict_data['bus_0_p_mw'].append(self.grid.res_bus.p_mw[0])
        self.dict_data['bus_1_p_mw'].append(self.grid.res_bus.p_mw[1])
        self.dict_data['bus_2_p_mw'].append(self.grid.res_bus.p_mw[2])
        self.dict_data['bus_3_p_mw'].append(self.grid.res_bus.p_mw[3])
        self.dict_data['bus_4_p_mw'].append(self.grid.res_bus.p_mw[4])
        self.dict_data['bus_5_p_mw'].append(self.grid.res_bus.p_mw[5])
        self.dict_data['bus_6_p_mw'].append(self.grid.res_bus.p_mw[6])
        self.dict_data['bus_7_p_mw'].append(self.grid.res_bus.p_mw[7])
        self.dict_data['bus_8_p_mw'].append(self.grid.res_bus.p_mw[8])
        self.dict_data['bus_9_p_mw'].append(self.grid.res_bus.p_mw[9])
        self.dict_data['bus_10_p_mw'].append(self.grid.res_bus.p_mw[10])
        self.dict_data['bus_11_p_mw'].append(self.grid.res_bus.p_mw[11])
        self.dict_data['bus_12_p_mw'].append(self.grid.res_bus.p_mw[12])
        self.dict_data['bus_13_p_mw'].append(self.grid.res_bus.p_mw[13])
        self.dict_data['bus_14_p_mw'].append(self.grid.res_bus.p_mw[14])

        self.dict_data['bus_0_q_mvar'].append(self.grid.res_bus.q_mvar[0])
        self.dict_data['bus_1_q_mvar'].append(self.grid.res_bus.q_mvar[1])
        self.dict_data['bus_2_q_mvar'].append(self.grid.res_bus.q_mvar[2])
        self.dict_data['bus_3_q_mvar'].append(self.grid.res_bus.q_mvar[3])
        self.dict_data['bus_4_q_mvar'].append(self.grid.res_bus.q_mvar[4])
        self.dict_data['bus_5_q_mvar'].append(self.grid.res_bus.q_mvar[5])
        self.dict_data['bus_6_q_mvar'].append(self.grid.res_bus.q_mvar[6])
        self.dict_data['bus_7_q_mvar'].append(self.grid.res_bus.q_mvar[7])
        self.dict_data['bus_8_q_mvar'].append(self.grid.res_bus.q_mvar[8])
        self.dict_data['bus_9_q_mvar'].append(self.grid.res_bus.q_mvar[9])
        self.dict_data['bus_10_q_mvar'].append(self.grid.res_bus.q_mvar[10])
        self.dict_data['bus_11_q_mvar'].append(self.grid.res_bus.q_mvar[11])
        self.dict_data['bus_12_q_mvar'].append(self.grid.res_bus.q_mvar[12])
        self.dict_data['bus_13_q_mvar'].append(self.grid.res_bus.q_mvar[13])
        self.dict_data['bus_14_q_mvar'].append(self.grid.res_bus.q_mvar[14])

        self.dict_data['switch_0_status'].append(self.grid.switch.closed.iloc[0])
        self.dict_data['switch_1_status'].append(self.grid.switch.closed.iloc[1])
        self.dict_data['switch_2_status'].append(self.grid.switch.closed.iloc[2])
        self.dict_data['switch_3_status'].append(self.grid.switch.closed.iloc[3])
        self.dict_data['switch_4_status'].append(self.grid.switch.closed.iloc[4])
        self.dict_data['switch_5_status'].append(self.grid.switch.closed.iloc[5])
        self.dict_data['switch_6_status'].append(self.grid.switch.closed.iloc[6])
        self.dict_data['switch_7_status'].append(self.grid.switch.closed.iloc[7])

    def get_simulation_data(self):
        sim_data = pd.DataFrame.from_dict(self.dict_data)
        return sim_data
