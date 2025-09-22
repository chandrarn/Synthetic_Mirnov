#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  1 16:10:25 2025
    Container for C-Mod Mirnov Geometric data for the Limiter Probes
    The geometric positions and orientations are correctly stored for the 
    low m/n BP probes in get_Cmod_Data
@author: rianc
"""
from header_Cmod import mds, np, json

def Mirnov_Geometry(shotno,debug=True):
    if int(str(shotno)[1:3]) > 10:
        phi = {
        
            'BP01_ABK': -30.9, 'BP02_ABK': -30.9, 'BP03_ABK': -30.8, 'BP04_ABK': -30.7,
        
            'BP05_ABK': -30.6, 'BP06_ABK': -30.6, 'BP07_ABK': -30.6, 'BP08_ABK': -30.6,
        
            'BP09_ABK': -30.7, 'BP10_ABK': -30.8, 'BP11_ABK': -30.9, 'BP12_ABK': -30.9,
        
            'BP13_ABK': -20.1, 'BP14_ABK': -20.1, 'BP15_ABK': -20.2, 'BP16_ABK': -20.3,
        
            'BP17_ABK': -20.4, 'BP18_ABK': -20.4, 'BP19_ABK': -20.4, 'BP20_ABK': -20.4,
        
            'BP21_ABK': -20.3, 'BP22_ABK': -20.2, 'BP23_ABK': -20.1, 'BP24_ABK': -20.1,
        
            'BP1T_ABK': -23.1, 'BP2T_ABK': -25.5, 'BP3T_ABK': -27.9, 'BP4T_ABK': -23.1,
        
            'BP5T_ABK': -25.5, 'BP6T_ABK': -27.9,
        
            'BP01_GHK': -232.2, 'BP02_GHK': -232.2, 'BP03_GHK': -232.1, 'BP04_GHK': -232,
        
            'BP05_GHK': -231.9, 'BP06_GHK': -231.9, 'BP07_GHK': -231.8, 'BP08_GHK': -231.7,
        
            'BP09_GHK': -231.8, 'BP10_GHK': -231.9, 'BP11_GHK': -231.9, 'BP12_GHK': -232,
        
            'BP13_GHK': -232.1, 'BP14_GHK': -232.2, 'BP15_GHK': -232.2,
        
            'BP16_GHK': -221.5, 'BP17_GHK': -221.5, 'BP18_GHK': -221.6, 'BP19_GHK': -221.6,
        
            'BP20_GHK': -221.7, 'BP21_GHK': -221.7, 'BP22_GHK': -221.8, 'BP23_GHK': -221.8,
        
            'BP24_GHK': -221.7, 'BP25_GHK': -221.7, 'BP26_GHK': -221.6, 'BP27_GHK': -221.6,
        
            'BP28_GHK': -221.5, 'BP29_GHK': -221.5,
        
            'BP1T_GHK': -224.4, 'BP2T_GHK': -226.8, 'BP3T_GHK': -229.2,
        
            'BP4T_GHK': -224.4, 'BP5T_GHK': -226.8, 'BP6T_GHK': -229.2,
        
            'BP_KA_TOP': -344.8, 'BP_KA_BOT': -344.8, 'BP_AB_TOP': -10.16,
        
            'BP_BC_TOP': -59.87, 'BP_BC_BOT': -59.87, 'BP_EF_TOP': -170.09, 'BP_EF_BOT': -170.09
        
        }
        
         
        
        Z = {
        
            'BP01_ABK': 0.226, 'BP02_ABK': 0.2, 'BP03_ABK': 0.169, 'BP04_ABK': 0.137,
        
            'BP05_ABK': 0.104, 'BP06_ABK': 0.076, 'BP07_ABK': -0.076, 'BP08_ABK': -0.104,
        
            'BP09_ABK': -0.137, 'BP10_ABK': -0.169, 'BP11_ABK': -0.2, 'BP12_ABK': -0.226,
        
            'BP13_ABK': 0.226, 'BP14_ABK': 0.2, 'BP15_ABK': 0.169, 'BP16_ABK': 0.137,
        
            'BP17_ABK': 0.104, 'BP18_ABK': 0.076, 'BP19_ABK': -0.076, 'BP20_ABK': -0.104,
        
            'BP21_ABK': -0.137, 'BP22_ABK': -0.169, 'BP23_ABK': -0.2, 'BP24_ABK': -0.226,
        
            'BP1T_ABK': 0.103, 'BP2T_ABK': 0.103, 'BP3T_ABK': 0.103, 'BP4T_ABK': -0.103,
        
            'BP5T_ABK': -0.103, 'BP6T_ABK': -0.103,
        
            'BP01_GHK': 0.227, 'BP02_GHK': 0.201, 'BP03_GHK': 0.17, 'BP04_GHK': 0.138,
        
            'BP05_GHK': 0.108, 'BP06_GHK': 0.077, 'BP07_GHK': 0.038, 'BP08_GHK': 0.0,
        
            'BP09_GHK': -0.038, 'BP10_GHK': -0.077, 'BP11_GHK': -0.108, 'BP12_GHK': -0.138,
        
            'BP13_GHK': -0.17, 'BP14_GHK': -0.201, 'BP15_GHK': -0.227,
        
            'BP16_GHK': 0.227, 'BP17_GHK': 0.201, 'BP18_GHK': 0.17, 'BP19_GHK': 0.138,
        
            'BP20_GHK': 0.108, 'BP21_GHK': 0.077, 'BP22_GHK': 0.038, 'BP23_GHK': -0.038,
        
            'BP24_GHK': -0.077, 'BP25_GHK': -0.108, 'BP26_GHK': -0.138, 'BP27_GHK': -0.17,
        
            'BP28_GHK': -0.201, 'BP29_GHK': -0.227,
        
            'BP1T_GHK': 0.1, 'BP2T_GHK': 0.1, 'BP3T_GHK': 0.1,
        
            'BP4T_GHK': -0.1, 'BP5T_GHK': -0.1, 'BP6T_GHK': -0.1,
        
            'BP_KA_TOP': 0.0985, 'BP_KA_BOT': -0.0985, 'BP_AB_TOP': 0.0985,
        
            'BP_BC_TOP': 0.0985, 'BP_BC_BOT': -0.0985, 'BP_EF_TOP': 0.0985, 'BP_EF_BOT': -0.0985
        
        }
        
         
        
        R = {
        
            'BP01_ABK': 0.8635, 'BP02_ABK': 0.8834, 'BP03_ABK': 0.9014, 'BP04_ABK': 0.9173,
        
            'BP05_ABK': 0.9293, 'BP06_ABK': 0.9362, 'BP07_ABK': 0.9362, 'BP08_ABK': 0.9293,
        
            'BP09_ABK': 0.9173, 'BP10_ABK': 0.9014, 'BP11_ABK': 0.8834, 'BP12_ABK': 0.8635,
        
            'BP13_ABK': 0.8635, 'BP14_ABK': 0.8834, 'BP15_ABK': 0.9014, 'BP16_ABK': 0.9173,
        
            'BP17_ABK': 0.9293, 'BP18_ABK': 0.9362, 'BP19_ABK': 0.9362, 'BP20_ABK': 0.9293,
        
            'BP21_ABK': 0.9173, 'BP22_ABK': 0.9014, 'BP23_ABK': 0.8834, 'BP24_ABK': 0.8635,
        
            'BP1T_ABK': 0.9045, 'BP2T_ABK': 0.9045, 'BP3T_ABK': 0.9045, 'BP4T_ABK': 0.9045,
        
            'BP5T_ABK': 0.9045, 'BP6T_ABK': 0.9045,
        
            'BP01_GHK': 0.8669, 'BP02_GHK': 0.8858, 'BP03_GHK': 0.9047, 'BP04_GHK': 0.9197,
        
            'BP05_GHK': 0.9306, 'BP06_GHK': 0.9396, 'BP07_GHK': 0.9456, 'BP08_GHK': 0.9476,
        
            'BP09_GHK': 0.9456, 'BP10_GHK': 0.9396, 'BP11_GHK': 0.9306, 'BP12_GHK': 0.9197,
        
            'BP13_GHK': 0.9047, 'BP14_GHK': 0.8858, 'BP15_GHK': 0.8669,
        
            'BP16_GHK': 0.8669, 'BP17_GHK': 0.8858, 'BP18_GHK': 0.9047, 'BP19_GHK': 0.9197,
        
            'BP20_GHK': 0.9306, 'BP21_GHK': 0.9396, 'BP22_GHK': 0.9456, 'BP23_GHK': 0.9456,
        
            'BP24_GHK': 0.9396, 'BP25_GHK': 0.9306, 'BP26_GHK': 0.9197, 'BP27_GHK': 0.9047,
        
            'BP28_GHK': 0.8858, 'BP29_GHK': 0.8669,
        
            'BP1T_GHK': 0.9042, 'BP2T_GHK': 0.9042, 'BP3T_GHK': 0.9042,
        
            'BP4T_GHK': 0.9042, 'BP5T_GHK': 0.9042, 'BP6T_GHK': 0.9042,
        
            'BP_KA_TOP': 0.9126, 'BP_KA_BOT': 0.9131, 'BP_AB_TOP': 0.9126,
        
            'BP_BC_TOP': 0.9146, 'BP_BC_BOT': 0.9151, 'BP_EF_TOP': 0.9126, 'BP_EF_BOT': 0.9131
        
        }
        
         
        
        theta_pol = {
        
            'BP01_ABK': 49.8, 'BP02_ABK': 43.6, 'BP03_ABK': 36.5, 'BP04_ABK': 29.3,
        
            'BP05_ABK': 22.1, 'BP06_ABK': 16.1, 'BP07_ABK': -16.1, 'BP08_ABK': -22.1,
        
            'BP09_ABK': -29.3, 'BP10_ABK': -36.5, 'BP11_ABK': -43.6, 'BP12_ABK': -49.8,
        
            'BP13_ABK': 49.8, 'BP14_ABK': 43.6, 'BP15_ABK': 36.5, 'BP16_ABK': 29.3,
        
            'BP17_ABK': 22.1, 'BP18_ABK': 16.1, 'BP19_ABK': -16.1, 'BP20_ABK': -22.1,
        
            'BP21_ABK': -29.3, 'BP22_ABK': -36.5, 'BP23_ABK': -43.6, 'BP24_ABK': -49.8,
        
            'BP1T_ABK': 23.7, 'BP2T_ABK': 23.7, 'BP3T_ABK': 23.7, 'BP4T_ABK': -23.7,
        
            'BP5T_ABK': -23.7, 'BP6T_ABK': -23.7,
        
            'BP01_GHK': 49.6, 'BP02_GHK': 43.4, 'BP03_GHK': 36.3, 'BP04_GHK': 29.3,
        
            'BP05_GHK': 22.8, 'BP06_GHK': 16.1, 'BP07_GHK': 7.9, 'BP08_GHK': 0.0,
        
            'BP09_GHK': -7.9, 'BP10_GHK': -16.1, 'BP11_GHK': -22.8, 'BP12_GHK': -29.3,
        
            'BP13_GHK': -36.3, 'BP14_GHK': -43.4, 'BP15_GHK': -49.6,
        
            'BP16_GHK': 49.6, 'BP17_GHK': 43.4, 'BP18_GHK': 36.3, 'BP19_GHK': 29.3,
        
            'BP20_GHK': 22.8, 'BP21_GHK': 16.1, 'BP22_GHK': 7.9, 'BP23_GHK': -7.9,
        
            'BP24_GHK': -16.1, 'BP25_GHK': -22.8, 'BP26_GHK': -29.3, 'BP27_GHK': -36.3,
        
            'BP28_GHK': -43.4, 'BP29_GHK': -49.6,
        
            'BP1T_GHK': 23.1, 'BP2T_GHK': 23.1, 'BP3T_GHK': 23.1,
        
            'BP4T_GHK': -23.1, 'BP5T_GHK': -23.1, 'BP6T_GHK': -23.1,
        
            'BP_KA_TOP': 18, 'BP_KA_BOT': -18, 'BP_AB_TOP': 18,
        
            'BP_BC_TOP': 18, 'BP_BC_BOT': -18, 'BP_EF_TOP': 18, 'BP_EF_BOT': -18
        
        }
        
    else:
        
        phi = {

            "BP01_ABK": -30.90, "BP02_ABK": -30.90, "BP03_ABK": -30.80, "BP04_ABK": -30.70,
        
            "BP05_ABK": -30.60, "BP06_ABK": -30.60, "BP07_ABK": -30.60, "BP08_ABK": -30.60,
        
            "BP09_ABK": -30.70, "BP11_ABK": -30.90, "BP12_ABK": -30.90, "BP20_ABK": -20.40,
        
            "BP1T_ABK": -23.10, "BP2T_ABK": -25.50, "BP3T_ABK": -27.90, "BP4T_ABK": -23.10,
        
            "BP5T_ABK": -25.50, "BP6T_ABK": -27.90, "BP02_GHK": -232.20, "BP04_GHK": -232.00,
        
            "BP05_GHK": -231.90, "BP06_GHK": -231.90, "BP07_GHK": -231.80, "BP08_GHK": -231.70,
        
            "BP10_GHK": -231.90, "BP11_GHK": -231.90, "BP12_GHK": -232.00, "BP13_GHK": -232.10,
        
            "BP14_GHK": -232.20, "BP19_GHK": -221.60, "BP24_GHK": -221.70, "BP26_GHK": -221.60,
        
            "BP28_GHK": -221.50, "BP1T_GHK": -224.40, "BP2T_GHK": -226.80, "BP3T_GHK": -229.20,
        
            "BP4T_GHK": -224.40, "BP5T_GHK": -226.80, "BP6T_GHK": -229.20, "BP_KA_TOP": -344.80,
        
            "BP_KA_BOT": -344.80, "BP_AB_TOP": -10.16, "BP_BC_TOP": -59.87, "BP_BC_BOT": -59.87,
        
            "BP_EF_TOP": -169.55, "BP_EF_BOT": -169.55
        
        }
        
         
        
        Z = {
        
            "BP01_ABK": 0.2260, "BP02_ABK": 0.2000, "BP03_ABK": 0.1690, "BP04_ABK": 0.1370,
        
            "BP05_ABK": 0.1040, "BP06_ABK": 0.0760, "BP07_ABK": -0.0760, "BP08_ABK": -0.1040,
        
            "BP09_ABK": -0.1370, "BP11_ABK": -0.2000, "BP12_ABK": -0.2260, "BP20_ABK": -0.1040,
        
            "BP1T_ABK": 0.1030, "BP2T_ABK": 0.1030, "BP3T_ABK": 0.1030, "BP4T_ABK": -0.1030,
        
            "BP5T_ABK": -0.1030, "BP6T_ABK": -0.1030, "BP02_GHK": 0.2010, "BP04_GHK": 0.1380,
        
            "BP05_GHK": 0.1080, "BP06_GHK": 0.0770, "BP07_GHK": 0.0380, "BP08_GHK": 0.0000,
        
            "BP10_GHK": -0.0770, "BP11_GHK": -0.1080, "BP12_GHK": -0.1380, "BP13_GHK": -0.1700,
        
            "BP14_GHK": -0.2010, "BP19_GHK": 0.1380, "BP24_GHK": -0.0770, "BP26_GHK": -0.1380,
        
            "BP28_GHK": -0.2010, "BP1T_GHK": 0.1000, "BP2T_GHK": 0.1000, "BP3T_GHK": 0.1000,
        
            "BP4T_GHK": -0.1000, "BP5T_GHK": -0.1000, "BP6T_GHK": -0.1000, "BP_KA_TOP": 0.0985,
        
            "BP_KA_BOT": -0.0985, "BP_AB_TOP": 0.0985, "BP_BC_TOP": 0.0985, "BP_BC_BOT": -0.0985,
        
            "BP_EF_TOP": 0.1082, "BP_EF_BOT": -0.1092
        
        }
        
         
        
        R = {
        
            "BP01_ABK": 0.8635, "BP02_ABK": 0.8834, "BP03_ABK": 0.9014, "BP04_ABK": 0.9173,
        
            "BP05_ABK": 0.9293, "BP06_ABK": 0.9362, "BP07_ABK": 0.9362, "BP08_ABK": 0.9293,
        
            "BP09_ABK": 0.9173, "BP11_ABK": 0.8834, "BP12_ABK": 0.8635, "BP20_ABK": 0.9293,
        
            "BP1T_ABK": 0.9045, "BP2T_ABK": 0.9045, "BP3T_ABK": 0.9045, "BP4T_ABK": 0.9045,
        
            "BP5T_ABK": 0.9045, "BP6T_ABK": 0.9045, "BP02_GHK": 0.8858, "BP04_GHK": 0.9197,
        
            "BP05_GHK": 0.9306, "BP06_GHK": 0.9396, "BP07_GHK": 0.9456, "BP08_GHK": 0.9476,
        
            "BP10_GHK": 0.9396, "BP11_GHK": 0.9306, "BP12_GHK": 0.9197, "BP13_GHK": 0.9047,
        
            "BP14_GHK": 0.8858, "BP19_GHK": 0.9197, "BP24_GHK": 0.9396, "BP26_GHK": 0.9197,
        
            "BP28_GHK": 0.8858, "BP1T_GHK": 0.9042, "BP2T_GHK": 0.9042, "BP3T_GHK": 0.9042,
        
            "BP4T_GHK": 0.9042, "BP5T_GHK": 0.9042, "BP6T_GHK": 0.9042, "BP_KA_TOP": 0.9126,
        
            "BP_KA_BOT": 0.9131, "BP_AB_TOP": 0.9126, "BP_BC_TOP": 0.9146, "BP_BC_BOT": 0.9151,
        
            "BP_EF_TOP": 0.9126, "BP_EF_BOT": 0.9131
        
        }
        
         
        
        theta_pol = {
        
            "BP01_ABK": 49.8, "BP02_ABK": 43.6, "BP03_ABK": 36.5, "BP04_ABK": 29.3,
        
            "BP05_ABK": 22.1, "BP06_ABK": 16.1, "BP07_ABK": -16.1, "BP08_ABK": -22.1,
        
            "BP09_ABK": -29.3, "BP11_ABK": -43.6, "BP12_ABK": -49.8, "BP20_ABK": -22.1,
        
            "BP1T_ABK": 23.7, "BP2T_ABK": 23.7, "BP3T_ABK": 23.7, "BP4T_ABK": -23.7,
        
            "BP5T_ABK": -23.7, "BP6T_ABK": -23.7, "BP02_GHK": 43.4, "BP04_GHK": 29.3,
        
            "BP05_GHK": 22.8, "BP06_GHK": 16.1, "BP07_GHK": 7.9, "BP08_GHK": 0.0,
        
            "BP10_GHK": -16.1, "BP11_GHK": -22.8, "BP12_GHK": -29.3, "BP13_GHK": -36.3,
        
            "BP14_GHK": -43.4, "BP19_GHK": 29.3, "BP24_GHK": -16.1, "BP26_GHK": -29.3,
        
            "BP28_GHK": -43.4, "BP1T_GHK": 23.1, "BP2T_GHK": 23.1, "BP3T_GHK": 23.1,
        
            "BP4T_GHK": -23.1, "BP5T_GHK": -23.1, "BP6T_GHK": -23.1, "BP_KA_TOP": 18.0,
        
            "BP_KA_BOT": -18.0, "BP_AB_TOP": 18.0, "BP_BC_TOP": 18.0, "BP_BC_BOT": -18.0,
        
            "BP_EF_TOP": 19.8, "BP_EF_BOT": -20.0
        
        }
        
    ##############################################
    # Correction for theta_pol: Currently, it's the _geometric_ theta, e.g. arctan(z,r)
    # This is redundant information, switching to theta_pol = sensor orientation
    # Node locations as per Magnetics page on Cmod wiki
    #
    try:
        # raise Exception('Force use of hardcoded values')
        conn = mds.Connection('alcdata')
        conn.openTree('cmod',shotno)
        theta_pol_ab = conn.get(r'\MAGNETICS::TOP.PROCESSED.RF_LIM_DATA:THETA_POL_AB').data()
        theta_pol_gh = conn.get(r'\MAGNETICS::TOP.PROCESSED.RF_LIM_DATA:THETA_POL_GH').data()
        nodenames = conn.get(r'\MAGNETICS::TOP.RF_LIM_COILS:NODENAME').data()
        if debug:
            print(f'Using MDSplus data for shot {shotno}')
    except:
        theta_pol_ab, theta_pol_gh, nodenames = hardcodedVals(shotno)
        if debug:
            print(f'Using hardcoded values for shot {shotno}')

    for sensor_name in theta_pol:
        
        try: sensor_index = int(np.argwhere(nodenames==sensor_name)[0,0])
        except:# will fail for _TOP, _BOT caseses. Use BP[3/8]T_AB_K as replacement, R,Z values are closest match
            if 'TOP' in sensor_name: sensor_index= int(np.argwhere(nodenames=='BP3T_ABK')[0,0])
            elif 'BOT' in sensor_name: sensor_index= int(np.argwhere(nodenames=='BP08_ABK')[0,0])
            else: raise SyntaxError(f'Sensor {sensor_name} not found in nodenames')
        
        theta_pol[sensor_name] = theta_pol_ab[sensor_index] if 'AB' in \
            sensor_name else theta_pol_gh[sensor_index-30]
    
    ###############
    # Correction for phi: rotate to match CAD: Overall shift of ~60deg, rel shift of 7
    for node_name in phi:
            if 'AB' in node_name: phase_offset= (149.489-88)
            else: phase_offset= (149.489-88-7.2) # This shift is not verified for the TOP/BOT sensors
            phi[node_name] += phase_offset
    ###############
    # Corrections to put limiter sensors inside of limiter
    for node_name in R:
        # Correction for _T sensors to be under tile face
        continue
        # The below code is depreciated: flat surface limiters underpredicts the signal
        if 'T' in node_name and 'O' not in node_name: R[node_name] += 0.01*0 
        ###############
         # Correction for limiter side sensors to be inside of limiter
        elif 'T' not in node_name:
            continue 
            # The below code is depreciated: The sensors are actually attached to the side of the limiter
            if 'AB' in node_name:
                # print('Check', int(node_name[2:4]))
                if int(node_name[2:4]) <=12: 
                    phi[node_name] +=  1
                else: phi[node_name] -= 1
            else: 
                if int(node_name[2:4]) <=15: phi[node_name] +=  1
                else: phi[node_name] -= 1
    ####################################
    with open('../C-Mod/C_Mod_Mirnov_Geometry_R.json','w', encoding='utf-8') as f: 
        json.dump(R,f, ensure_ascii=False, indent=4)
    with open('../C-Mod/C_Mod_Mirnov_Geometry_Z.json','w', encoding='utf-8') as f: 
        json.dump(Z,f, ensure_ascii=False, indent=4)
    with open('../C-Mod/C_Mod_Mirnov_Geometry_Phi.json','w', encoding='utf-8') as f: 
        json.dump(phi,f, ensure_ascii=False, indent=4)
    ####################################
    return phi, theta_pol, R, Z


######################################################
def hardcodedVals(shotno):
    theta_pol_ab = [ -50.6,  -55.6,  -61.5,  -67.3,  -72.7,  -77.9, -102.1, -107.1,
                    -112.9, -118.6, -124.3, -129.5,  -50.6,  -55.6,  -61.5,  -67.3,
                    -72.7,  -77.9, -102.1, -107.1, -112.9, -118.6, -124.3, -129.5,
                    -71.1,  -71.1,  -71.1, -108.9, -108.9, -108.9]
    theta_pol_gh = [ -50.5,  -55.7,  -61.5,  -67.2,  -72.9,  -77.6,  -84.4,  -90. ,
                    -96.1, -102.4, -107.3, -112.7, -118.5, -124.4, -129.4,  -50.5,
                    -55.7,  -61.5,  -67.2,  -72.9,  -77.6,  -84.4,  -96.1, -102.4,
                    -107.3, -112.7, -118.5, -124.4, -129.4,  -72.3,  -72.3,  -72.3,
                    -107.7, -107.7, -107.7]
    nodenames = ['BP01_ABK', 'BP02_ABK', 'BP03_ABK', 'BP04_ABK', 'BP05_ABK',
                'BP06_ABK', 'BP07_ABK', 'BP08_ABK', 'BP09_ABK', 'BP10_ABK',
                'BP11_ABK', 'BP12_ABK', 'BP13_ABK', 'BP14_ABK', 'BP15_ABK',
                'BP16_ABK', 'BP17_ABK', 'BP18_ABK', 'BP19_ABK', 'BP20_ABK',
                'BP21_ABK', 'BP22_ABK', 'BP23_ABK', 'BP24_ABK', 'BP1T_ABK',
                'BP2T_ABK', 'BP3T_ABK', 'BP4T_ABK', 'BP5T_ABK', 'BP6T_ABK',
                'BP01_GHK', 'BP02_GHK', 'BP03_GHK', 'BP04_GHK', 'BP05_GHK',
                'BP06_GHK', 'BP07_GHK', 'BP08_GHK', 'BP09_GHK', 'BP10_GHK',
                'BP11_GHK', 'BP12_GHK', 'BP13_GHK', 'BP14_GHK', 'BP15_GHK',
                'BP16_GHK', 'BP17_GHK', 'BP18_GHK', 'BP19_GHK', 'BP20_GHK',
                'BP21_GHK', 'BP22_GHK', 'BP23_GHK', 'BP24_GHK', 'BP25_GHK',
                'BP26_GHK', 'BP27_GHK', 'BP28_GHK', 'BP29_GHK', 'BP1T_GHK',
                'BP2T_GHK', 'BP3T_GHK', 'BP4T_GHK', 'BP5T_GHK', 'BP6T_GHK',
                'BP01_K', 'BP02_K', 'BP03_K', 'BP04_K', 'BP05_K', 'BP06_K']
    
    nodenames = np.array(nodenames)
    return theta_pol_ab, theta_pol_gh, nodenames