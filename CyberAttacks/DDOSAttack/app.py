# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# from sklearn.preprocessing import MinMaxScaler

# # Load the trained model
# @st.cache_resource
# def load_model():
#     model_path = r"C:\Users\Paras\Desktop\IDSFedSSL\CyberAttacks\DDOSAttack\ddos.pkl"
#     with open(model_path, 'rb') as f:
#         model = pickle.load(f)
#     return model

# # Function to preprocess input data
# def preprocess_input(input_data):
#     # Convert to DataFrame
#     input_df = pd.DataFrame([input_data])
    
#     # Define numerical features for scaling (based on dataset)
#     numerical_features = ['pktcount', 'bytecount', 'dur', 'dur_nsec', 'tot_dur', 'flows', 
#                          'packetins', 'pktperflow', 'byteperflow', 'pktrate', 'Pairflow', 
#                          'port_no', 'tx_bytes', 'rx_bytes', 'tx_kbps', 'rx_kbps', 'tot_kbps']
    
#     # Initialize and fit scaler (this would ideally be pre-fitted, but we'll do it here)
#     scaler = MinMaxScaler()
    
#     # For demo purposes, we'll scale with min=0 and max=2*input_value
#     # In a real app, you should use the same scaler used during training
#     min_values = {feature: 0 for feature in numerical_features}
#     max_values = {feature: 2*input_data[feature] if input_data[feature] != 0 else 1 
#                   for feature in numerical_features}
    
#     for feature in numerical_features:
#         input_df[feature] = (input_df[feature] - min_values[feature]) / (
#             max_values[feature] - min_values[feature])
    
#     # Encode categorical features
#     protocol_mapping = {'UDP': 0, 'TCP': 1, 'ICMP': 2}
#     input_df['Protocol'] = input_df['Protocol'].map(protocol_mapping)
    
#     return input_df

# def main():
#     st.title("DDoS Attack Detection in SDN Networks")
#     st.write("""
#     This application uses a machine learning model to detect potential DDoS attacks 
#     based on network traffic features in Software Defined Networks (SDN).
#     """)
    
#     # Load model
#     try:
#         model = load_model()
#     except Exception as e:
#         st.error(f"Could not load model file. Error: {str(e)}")
#         st.error("Please ensure the model file exists at the specified path.")
#         return
    
#     # Sidebar for user inputs
#     st.sidebar.header("Input Network Traffic Parameters")
    
#     # Create input fields for all relevant features
#     protocol = st.sidebar.selectbox("Protocol", ['UDP', 'TCP', 'ICMP'])
#     pktcount = st.sidebar.number_input("Packet Count", min_value=0, value=45304)
#     bytecount = st.sidebar.number_input("Byte Count", min_value=0, value=48294064)
#     dur = st.sidebar.number_input("Duration (sec)", min_value=0, value=100)
#     dur_nsec = st.sidebar.number_input("Duration (nsec)", min_value=0, value=716000000)
#     tot_dur = st.sidebar.number_input("Total Duration (nsec)", min_value=0.0, value=1.01e11)
#     flows = st.sidebar.number_input("Number of Flows", min_value=0, value=3)
#     packetins = st.sidebar.number_input("Packet Ins", min_value=0, value=5200)
#     pktperflow = st.sidebar.number_input("Packets per Flow", min_value=0, value=6381)
#     byteperflow = st.sidebar.number_input("Bytes per Flow", min_value=0, value=4716150)
#     pktrate = st.sidebar.number_input("Packet Rate", min_value=0, value=212)
#     Pairflow = st.sidebar.number_input("Pair Flow", min_value=0, value=0)
#     port_no = st.sidebar.number_input("Port Number", min_value=1, max_value=5, value=3)
#     tx_bytes = st.sidebar.number_input("TX Bytes", min_value=0, value=93252640)
#     rx_bytes = st.sidebar.number_input("RX Bytes", min_value=0, value=93280390)
#     tx_kbps = st.sidebar.number_input("TX Kbps", min_value=0.0, value=998.9)
#     rx_kbps = st.sidebar.number_input("RX Kbps", min_value=0.0, value=1003.81)
#     tot_kbps = st.sidebar.number_input("Total Kbps", min_value=0.0, value=2007.58)
    
#     # Create dictionary of input data
#     input_data = {
#         'Protocol': protocol,
#         'pktcount': pktcount,
#         'bytecount': bytecount,
#         'dur': dur,
#         'dur_nsec': dur_nsec,
#         'tot_dur': tot_dur,
#         'flows': flows,
#         'packetins': packetins,
#         'pktperflow': pktperflow,
#         'byteperflow': byteperflow,
#         'pktrate': pktrate,
#         'Pairflow': Pairflow,
#         'port_no': port_no,
#         'tx_bytes': tx_bytes,
#         'rx_bytes': rx_bytes,
#         'tx_kbps': tx_kbps,
#         'rx_kbps': rx_kbps,
#         'tot_kbps': tot_kbps
#     }
    
#     # Preprocess the input
#     processed_input = preprocess_input(input_data)
    
#     # Make prediction
#     if st.sidebar.button("Detect DDoS"):
#         try:
#             prediction = model.predict(processed_input)
            
#             # Try to get prediction probabilities if available
#             try:
#                 prediction_proba = model.predict_proba(processed_input)
#                 confidence = prediction_proba[0][1]*100 if prediction[0] == 1 else prediction_proba[0][0]*100
#             except:
#                 confidence = "N/A"
            
#             st.subheader("Prediction Results")
            
#             if prediction[0] == 1:
#                 st.error(" DDoS Attack Detected!")
#                 st.write(f"Confidence: {confidence}%")
                
#                 # Display attack characteristics
#                 st.subheader("Attack Characteristics")
                
#                 # Highlight suspicious values (example thresholds)
#                 suspicious_features = []
                
#                 if pktcount > 100000:
#                     suspicious_features.append(f"High packet count: {pktcount}")
#                 if bytecount > 10000000:
#                     suspicious_features.append(f"High byte count: {bytecount}")
#                 if pktrate > 300:
#                     suspicious_features.append(f"High packet rate: {pktrate}")
#                 if flows > 10:
#                     suspicious_features.append(f"Unusually high number of flows: {flows}")
                
#                 if suspicious_features:
#                     st.write("Suspicious traffic patterns detected:")
#                     for feature in suspicious_features:
#                         st.write(f"- {feature}")
#                 else:
#                     st.write("No specific suspicious patterns identified beyond model prediction.")
                
#             else:
#                 st.success(" Normal Traffic")
#                 st.write(f"Confidence: {confidence}%")
            
#         except Exception as e:
#             st.error(f"Error making prediction: {str(e)}")
    
#     # Main section for explanation and model info
#     st.subheader("About the Model")
#     st.write("""
#     This model analyzes network traffic patterns in SDN environments to detect potential 
#     DDoS attacks. It examines features like packet counts, flow durations, and traffic rates.
    
#     **Model Input Features:**
#     - Protocol type (UDP/TCP/ICMP)
#     - Packet and byte counts
#     - Flow durations and rates
#     - Port numbers and traffic statistics
#     """)
    
#     st.subheader("How to Use")
#     st.write("""
#     1. Adjust the input parameters in the sidebar to match your network traffic
#     2. Click 'Detect DDoS' to run the detection
#     3. Review the prediction and confidence score
#     4. Examine any suspicious traffic patterns identified
#     """)
    
#     st.subheader("Sample Attack Indicators")
#     st.write("""
#     The following patterns might indicate a DDoS attack:
#     - Extremely high packet rates (>300 packets/sec)
#     - Large byte counts (>10MB) in short durations
#     - Multiple flows targeting the same destination port
#     - Unusually high traffic asymmetry (TX vs RX)
#     - Sudden spikes in packet counts (>100,000)
#     """)

# if __name__ == '__main__':
#     main()

# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# import requests
# from urllib.parse import urlparse
# import socket
# from sklearn.preprocessing import MinMaxScaler

# # Load the trained model
# @st.cache_resource
# def load_model():
#     model_path = r"C:\Users\Paras\Desktop\IDSFedSSL\CyberAttacks\DDOSAttack\ddos.pkl"
#     with open(model_path, 'rb') as f:
#         model = pickle.load(f)
#     return model

# def extract_url_features(url):
#     """Extract network features from a URL"""
#     try:
#         parsed = urlparse(url)
#         domain = parsed.netloc
#         ip = socket.gethostbyname(domain)
        
#         # Simulate traffic measurements (in a real app, you'd monitor actual traffic)
#         features = {
#             'Protocol': 'TCP',  # Assume HTTP/HTTPS
#             'pktcount': 1000 if "attack" not in url.lower() else 500000,
#             'bytecount': 100000 if "attack" not in url.lower() else 50000000,
#             'dur': 60,
#             'dur_nsec': 600000000,
#             'tot_dur': 60000000000,
#             'flows': 5 if "attack" not in url.lower() else 5000,
#             'packetins': 1000,
#             'pktperflow': 300 if "attack" not in url.lower() else 10,
#             'byteperflow': 30000,
#             'pktrate': 25 if "attack" not in url.lower() else 50000,
#             'Pairflow': 1,
#             'port_no': 443 if parsed.scheme == 'https' else 80,
#             'tx_bytes': 500000,
#             'rx_bytes': 500000 if "attack" not in url.lower() else 1000,
#             'tx_kbps': 50 if "attack" not in url.lower() else 5000,
#             'rx_kbps': 50,
#             'tot_kbps': 100 if "attack" not in url.lower() else 5000
#         }
#         return features, ip
#     except Exception as e:
#         st.error(f"Error processing URL: {str(e)}")
#         return None, None

# def preprocess_input(input_data):
#     """Preprocess features for model prediction"""
#     input_df = pd.DataFrame([input_data])
    
#     numerical_features = ['pktcount', 'bytecount', 'dur', 'dur_nsec', 'tot_dur', 'flows', 
#                          'packetins', 'pktperflow', 'byteperflow', 'pktrate', 'Pairflow', 
#                          'port_no', 'tx_bytes', 'rx_bytes', 'tx_kbps', 'rx_kbps', 'tot_kbps']
    
#     # Simple scaling (replace with your actual scaler if available)
#     for feature in numerical_features:
#         input_df[feature] = input_df[feature] / (input_df[feature].max() + 1e-10)
    
#     protocol_mapping = {'UDP': 0, 'TCP': 1, 'ICMP': 2}
#     input_df['Protocol'] = input_df['Protocol'].map(protocol_mapping)
    
#     return input_df

# def main():
#     st.title("URL-Based DDoS Attack Detection")
#     st.write("Analyze web traffic patterns for potential DDoS attacks")
    
#     model = load_model()
    
#     # Input URL
#     url = st.text_input("Enter URL to analyze:", "https://outlook.office.com/")
    
#     if st.button("Analyze URL"):
#         if not url.startswith(('http://', 'https://')):
#             url = 'http://' + url
            
#         features, ip = extract_url_features(url)
        
#         if features:
#             st.subheader("URL Analysis Results")
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.metric("Domain", urlparse(url).netloc)
#                 st.metric("IP Address", ip)
#                 st.metric("Protocol", "HTTPS" if url.startswith('https') else "HTTP")
            
#             with col2:
#                 st.metric("Port", 443 if url.startswith('https') else 80)
#                 st.metric("Path", urlparse(url).path)
            
#             # Preprocess and predict
#             processed = preprocess_input(features)
#             prediction = model.predict(processed)
            
#             if prediction[0] == 1:
#                 st.error(" Potential DDoS Attack Detected!")
                
#                 st.subheader("Attack Indicators Found:")
#                 indicators = []
#                 if features['pktcount'] > 100000:
#                     indicators.append(f"High packet count ({features['pktcount']})")
#                 if features['bytecount'] > 10000000:
#                     indicators.append(f"Large byte volume ({features['bytecount']} bytes)")
#                 if features['pktrate'] > 1000:
#                     indicators.append(f"Abnormal packet rate ({features['pktrate']} pkt/sec)")
#                 if features['flows'] > 1000:
#                     indicators.append(f"Excessive flows ({features['flows']})")
                
#                 if indicators:
#                     for indicator in indicators:
#                         st.write(f"- {indicator}")
#                 else:
#                     st.write("Model detected attack but no specific patterns identified")
#             else:
#                 st.success(" Normal Traffic Patterns")
                
#             # Show raw features (for debugging)
#             with st.expander("Show extracted features"):
#                 st.json(features)

# if __name__ == '__main__':
#     main()

# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# import time
# from scapy.all import sniff, IP, TCP, UDP, ICMP
# import psutil
# from threading import Thread, Event
# from collections import defaultdict
# from sklearn.preprocessing import MinMaxScaler

# # Global variables for traffic monitoring
# traffic_stats = defaultdict(lambda: {
#     'pktcount': 0,
#     'bytecount': 0,
#     'flows': set(),
#     'start_time': time.time(),
#     'protocols': defaultdict(int),
#     'ports': defaultdict(int)
# })

# stop_sniffing = Event()

# # Load the trained model
# @st.cache_resource
# def load_model():
#     model_path = r"C:\Users\Paras\Desktop\IDSFedSSL\CyberAttacks\DDOSAttack\ddos.pkl"
#     with open(model_path, 'rb') as f:
#         model = pickle.load(f)
#     return model

# def packet_handler(packet):
#     """Process each network packet"""
#     if IP in packet:
#         src = packet[IP].src
#         dst = packet[IP].dst
#         flow_key = (src, dst)
        
#         # Update general stats
#         traffic_stats['total']['pktcount'] += 1
#         traffic_stats['total']['bytecount'] += len(packet)
#         traffic_stats['total']['flows'].add(flow_key)
        
#         # Update protocol-specific stats
#         if TCP in packet:
#             protocol = 'TCP'
#             port = packet[TCP].dport
#         elif UDP in packet:
#             protocol = 'UDP'
#             port = packet[UDP].dport
#         elif ICMP in packet:
#             protocol = 'ICMP'
#             port = 0
#         else:
#             protocol = 'OTHER'
#             port = 0
            
#         traffic_stats['total']['protocols'][protocol] += 1
#         if port > 0:
#             traffic_stats['total']['ports'][port] += 1

# def start_sniffing():
#     """Start packet capture in background"""
#     sniff(prn=packet_handler, store=0, stop_filter=lambda _: stop_sniffing.is_set())

# def get_traffic_features():
#     """Convert captured stats to model features"""
#     duration = time.time() - traffic_stats['total']['start_time']
#     if duration == 0:
#         duration = 1  # prevent division by zero
    
#     total = traffic_stats['total']
#     protocols = total['protocols']
    
#     features = {
#         'Protocol': max(protocols.items(), key=lambda x: x[1])[0] if protocols else 'TCP',
#         'pktcount': total['pktcount'],
#         'bytecount': total['bytecount'],
#         'dur': duration,
#         'dur_nsec': duration * 1e9,
#         'tot_dur': duration * 1e9,
#         'flows': len(total['flows']),
#         'packetins': total['pktcount'],
#         'pktperflow': total['pktcount'] / max(1, len(total['flows'])),
#         'byteperflow': total['bytecount'] / max(1, len(total['flows'])),
#         'pktrate': total['pktcount'] / duration,
#         'Pairflow': 1 if len(total['flows']) < 100 else 0,  # Simple heuristic
#         'port_no': max(total['ports'].items(), key=lambda x: x[1])[0] if total['ports'] else 80,
#         'tx_bytes': total['bytecount'],
#         'rx_bytes': total['bytecount'],  # Simplified for demo
#         'tx_kbps': (total['bytecount'] * 8 / 1000) / duration,
#         'rx_kbps': (total['bytecount'] * 8 / 1000) / duration,
#         'tot_kbps': (total['bytecount'] * 8 / 1000) / duration * 2
#     }
#     return features

# def preprocess_input(input_data):
#     """Prepare features for model prediction"""
#     input_df = pd.DataFrame([input_data])
    
#     numerical_features = ['pktcount', 'bytecount', 'dur', 'dur_nsec', 'tot_dur', 'flows', 
#                          'packetins', 'pktperflow', 'byteperflow', 'pktrate', 'Pairflow', 
#                          'port_no', 'tx_bytes', 'rx_bytes', 'tx_kbps', 'rx_kbps', 'tot_kbps']
    
#     # Simple scaling (replace with your actual scaler)
#     for feature in numerical_features:
#         input_df[feature] = input_df[feature] / (input_df[feature].max() + 1e-10)
    
#     protocol_mapping = {'UDP': 0, 'TCP': 1, 'ICMP': 2, 'OTHER': 1}
#     input_df['Protocol'] = input_df['Protocol'].map(protocol_mapping)
    
#     return input_df

# def main():
#     st.title("Real-Time DDoS Detection System")
#     st.write("Monitoring network traffic for potential attacks")
    
#     model = load_model()
    
#     # Initialize session state
#     if 'monitoring' not in st.session_state:
#         st.session_state.monitoring = False
#         st.session_state.start_time = None
    
#     # Control panel
#     col1, col2 = st.columns(2)
#     with col1:
#         if st.button("Start Monitoring") and not st.session_state.monitoring:
#             st.session_state.monitoring = True
#             st.session_state.start_time = time.time()
#             stop_sniffing.clear()
#             Thread(target=start_sniffing, daemon=True).start()
#             st.rerun()
    
#     with col2:
#         if st.button("Stop Monitoring") and st.session_state.monitoring:
#             st.session_state.monitoring = False
#             stop_sniffing.set()
#             st.rerun()
    
#     # Display monitoring status
#     status_placeholder = st.empty()
#     chart_placeholder = st.empty()
#     alert_placeholder = st.empty()
#     stats_placeholder = st.empty()
    
#     # Main monitoring loop
#     while st.session_state.monitoring:
#         # Get current traffic features
#         features = get_traffic_features()
#         processed = preprocess_input(features)
        
#         # Make prediction
#         prediction = model.predict(processed)
        
#         # Update status display
#         with status_placeholder.container():
#             st.subheader("Current Status")
#             cols = st.columns(4)
#             cols[0].metric("Packets/s", f"{features['pktrate']:.1f}")
#             cols[1].metric("Traffic", f"{features['tot_kbps']:.1f} kbps")
#             cols[2].metric("Flows", features['flows'])
#             cols[3].metric("Status", " ATTACK" if prediction[0] == 1 else " Normal")
        
#         # Update traffic chart
#         with chart_placeholder.container():
#             st.subheader("Traffic Patterns")
#             chart_data = pd.DataFrame({
#                 'Metric': ['Packet Rate', 'Traffic Volume', 'Active Flows'],
#                 'Value': [features['pktrate'], features['tot_kbps'], features['flows']]
#             })
#             st.bar_chart(chart_data.set_index('Metric'))
        
#         # Show alerts if attack detected
#         if prediction[0] == 1:
#             with alert_placeholder.container():
#                 st.error("## DDoS Attack Detected!")
#                 indicators = []
#                 if features['pktrate'] > 1000:
#                     indicators.append(f"High packet rate ({features['pktrate']:.1f} pkts/sec)")
#                 if features['tot_kbps'] > 1000:
#                     indicators.append(f"High traffic volume ({features['tot_kbps']:.1f} kbps)")
#                 if features['flows'] > 500:
#                     indicators.append(f"Excessive flows ({features['flows']})")
#                 if features['Protocol'] == 'UDP' and features['pktrate'] > 500:
#                     indicators.append("UDP flood pattern detected")
                
#                 for indicator in indicators:
#                     st.write(f"- {indicator}")
        
#         # Show detailed stats
#         with stats_placeholder.container():
#             with st.expander("Detailed Traffic Statistics"):
#                 st.json(features)
        
#         time.sleep(1)  # Update interval
    
#     # Show final report when stopped
#     if not st.session_state.monitoring and st.session_state.start_time:
#         duration = time.time() - st.session_state.start_time
#         st.success(f"Monitoring completed after {duration:.1f} seconds")
#         if traffic_stats['total']['pktcount'] > 0:
#             st.json(get_traffic_features())

# if __name__ == '__main__':
#     main()

# 

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
from scapy.all import sniff, IP, TCP, UDP, ICMP
from threading import Thread, Event
from collections import defaultdict
import psutil
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Global variables for traffic monitoring
traffic_stats = defaultdict(lambda: {
    'pktcount': 0,
    'bytecount': 0,
    'flows': set(),
    'start_time': time.time(),
    'protocols': defaultdict(int),
    'ports': defaultdict(int),
    'dst_ips': defaultdict(int)
})

stop_sniffing = Event()

# Load the trained model and scaler
@st.cache_resource
def load_artifacts():
    model_path = r"C:\Users\Paras\Desktop\IDSFedSSL\CyberAttacks\DDOSAttack\ddos.pkl"
    scaler_path = r"C:\Users\Paras\Desktop\IDSFedSSL\CyberAttacks\DDOSAttack\scaler.pkl"
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    return model, scaler

def packet_handler(packet):
    """Process each network packet"""
    if IP in packet:
        src = packet[IP].src
        dst = packet[IP].dst
        proto = packet[IP].proto
        flow_key = (src, dst, proto)
        
        # Update general stats
        traffic_stats['total']['pktcount'] += 1
        traffic_stats['total']['bytecount'] += len(packet)
        traffic_stats['total']['flows'].add(flow_key)
        traffic_stats['total']['dst_ips'][dst] += 1
        
        # Update protocol-specific stats
        if TCP in packet:
            protocol = 'TCP'
            port = packet[TCP].dport
        elif UDP in packet:
            protocol = 'UDP'
            port = packet[UDP].dport
        elif ICMP in packet:
            protocol = 'ICMP'
            port = 0
        else:
            protocol = 'OTHER'
            port = 0
            
        traffic_stats['total']['protocols'][protocol] += 1
        if port > 0:
            traffic_stats['total']['ports'][port] += 1

def start_sniffing(interface=None):
    """Start packet capture in background"""
    try:
        sniff(prn=packet_handler, store=0, 
              stop_filter=lambda _: stop_sniffing.is_set(),
              iface=interface)
    except Exception as e:
        st.error(f"Packet capture error: {str(e)}")
        stop_sniffing.set()

def get_traffic_features():
    """Convert captured stats to model features"""
    duration = max(time.time() - traffic_stats['total']['start_time'], 1)
    total = traffic_stats['total']
    protocols = total['protocols']
    
    # Protocol features (one-hot encoded)
    protocol_features = {
        'Protocol_TCP': 1 if 'TCP' in protocols else 0,
        'Protocol_UDP': 1 if 'UDP' in protocols else 0,
        'Protocol_ICMP': 1 if 'ICMP' in protocols else 0,
        'Protocol_OTHER': 1 if any(p not in ['TCP', 'UDP', 'ICMP'] for p in protocols) else 0
    }
    
    # Destination IP features (using top 5 detected IPs)
    top_dst_ips = sorted(total['dst_ips'].items(), key=lambda x: x[1], reverse=True)[:5]
    dst_features = {
        f'dst_{ip}': count for ip, count in top_dst_ips
    }
    
    # Add default IPs if not present
    for ip in ['10.0.0.10', '10.0.0.11', '10.0.0.12']:
        if f'dst_{ip}' not in dst_features:
            dst_features[f'dst_{ip}'] = 0
    
    # Get most common port
    main_port = max(total['ports'].items(), key=lambda x: x[1])[0] if total['ports'] else 80
    
    # Combine all features
    features = {
        'pktcount': total['pktcount'],
        'bytecount': total['bytecount'],
        'dur': duration,
        'dur_nsec': duration * 1e9,
        'tot_dur': duration * 1e9,
        'flows': len(total['flows']),
        'packetins': total['pktcount'],
        'pktperflow': total['pktcount'] / max(1, len(total['flows'])),
        'byteperflow': total['bytecount'] / max(1, len(total['flows'])),
        'pktrate': total['pktcount'] / duration,
        'Pairflow': 1 if len(total['flows']) < 100 else 0,
        'port_no': main_port,
        'tx_bytes': total['bytecount'],
        'rx_bytes': total['bytecount'],
        'tx_kbps': (total['bytecount'] * 8 / 1000) / duration,
        'rx_kbps': (total['bytecount'] * 8 / 1000) / duration,
        'tot_kbps': (total['bytecount'] * 8 / 1000) / duration * 2,
        **protocol_features,
        **dst_features
    }
    
    return features

def main():
    st.title("Real-Time DDoS Detection System")
    st.write("Monitoring network traffic for potential attacks")
    
    # Load model and scaler
    try:
        model, scaler = load_artifacts()
    except Exception as e:
        st.error(f"Failed to load model/scaler: {str(e)}")
        return
    
    # Get available network interfaces
    interfaces = psutil.net_if_stats().keys()
    selected_iface = st.selectbox("Select Network Interface", list(interfaces))
    
    # Initialize session state
    if 'monitoring' not in st.session_state:
        st.session_state.monitoring = False
        st.session_state.start_time = None
    
    # Control panel
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Monitoring") and not st.session_state.monitoring:
            st.session_state.monitoring = True
            st.session_state.start_time = time.time()
            traffic_stats['total'] = {  # Reset stats
                'pktcount': 0,
                'bytecount': 0,
                'flows': set(),
                'start_time': time.time(),
                'protocols': defaultdict(int),
                'ports': defaultdict(int),
                'dst_ips': defaultdict(int)
            }
            stop_sniffing.clear()
            Thread(target=start_sniffing, args=(selected_iface,), daemon=True).start()
            st.rerun()
    
    with col2:
        if st.button("Stop Monitoring") and st.session_state.monitoring:
            st.session_state.monitoring = False
            stop_sniffing.set()
            st.rerun()
    
    # Display monitoring status
    status_placeholder = st.empty()
    chart_placeholder = st.empty()
    alert_placeholder = st.empty()
    stats_placeholder = st.empty()
    
    # Main monitoring loop
    while st.session_state.monitoring:
        try:
            # Get current traffic features
            features = get_traffic_features()
            
            # Prepare DataFrame for scaling
            input_df = pd.DataFrame([features])
            
            # Ensure all expected columns are present
            expected_columns = scaler.feature_names_in_
            for col in expected_columns:
                if col not in input_df.columns:
                    input_df[col] = 0  # Add missing columns with default value
            
            # Reorder columns to match scaler expectations
            input_df = input_df[expected_columns]
            
            # Scale features
            input_df_scaled = scaler.transform(input_df)
            
            # Make prediction
            prediction = model.predict(input_df_scaled)
            try:
                prediction_proba = model.predict_proba(input_df_scaled)
                confidence = prediction_proba[0][1] if prediction[0] == 1 else prediction_proba[0][0]
            except:
                confidence = 1.0
            
            # Update status display
            with status_placeholder.container():
                st.subheader("Current Status")
                cols = st.columns(4)
                cols[0].metric("Packets/s", f"{features['pktrate']:.1f}")
                cols[1].metric("Traffic", f"{features['tot_kbps']:.1f} kbps")
                cols[2].metric("Flows", features['flows'])
                status_text = "⚠️ ATTACK" if prediction[0] == 1 else "✅ Normal"
                cols[3].metric("Status", f"{status_text} ({confidence*100:.1f}%)")
            
            # Update traffic chart
            with chart_placeholder.container():
                st.subheader("Traffic Patterns")
                chart_data = pd.DataFrame({
                    'Metric': ['Packet Rate', 'Traffic Volume', 'Active Flows'],
                    'Value': [features['pktrate'], features['tot_kbps'], features['flows']]
                })
                st.bar_chart(chart_data.set_index('Metric'))
            
            # Show alerts if attack detected
            if prediction[0] == 1:
                with alert_placeholder.container():
                    st.error("## DDoS Attack Detected!")
                    
                    indicators = []
                    if features['pktrate'] > 1000:
                        indicators.append(f"High packet rate ({features['pktrate']:.1f} pkts/sec)")
                    if features['tot_kbps'] > 1000:
                        indicators.append(f"High traffic volume ({features['tot_kbps']:.1f} kbps)")
                    if features['flows'] > 500:
                        indicators.append(f"Excessive flows ({features['flows']})")
                    if features['Protocol_UDP'] == 1 and features['pktrate'] > 500:
                        indicators.append("UDP flood pattern detected")
                    
                    for indicator in indicators:
                        st.write(f"- {indicator}")
                    
                    st.write(f"Confidence: {confidence*100:.1f}%")
            
            # Show detailed stats
            with stats_placeholder.container():
                with st.expander("Detailed Traffic Statistics"):
                    st.json(features)
            
            time.sleep(1)  # Update interval
        
        except Exception as e:
            st.error(f"Error during monitoring: {str(e)}")
            st.session_state.monitoring = False
            stop_sniffing.set()
            break
    
    # Show final report when stopped
    if not st.session_state.monitoring and st.session_state.start_time:
        duration = time.time() - st.session_state.start_time
        st.success(f"Monitoring completed after {duration:.1f} seconds")
        
        if traffic_stats['total']['pktcount'] > 0:
            final_features = get_traffic_features()
            st.subheader("Final Traffic Summary")
            
            # Protocol distribution
            protocols = traffic_stats['total']['protocols']
            if protocols:
                st.write("### Protocol Distribution")
                proto_df = pd.DataFrame({
                    'Protocol': list(protocols.keys()),
                    'Count': list(protocols.values())
                })
                st.bar_chart(proto_df.set_index('Protocol'))
            
            # Port distribution
            ports = traffic_stats['total']['ports']
            if ports:
                st.write("### Top Destination Ports")
                port_df = pd.DataFrame({
                    'Port': list(ports.keys()),
                    'Count': list(ports.values())
                }).sort_values('Count', ascending=False).head(10)
                st.bar_chart(port_df.set_index('Port'))
            
            # Destination IP distribution
            dst_ips = traffic_stats['total']['dst_ips']
            if dst_ips:
                st.write("### Top Destination IPs")
                ip_df = pd.DataFrame({
                    'IP Address': list(dst_ips.keys()),
                    'Count': list(dst_ips.values())
                }).sort_values('Count', ascending=False).head(10)
                st.bar_chart(ip_df.set_index('IP Address'))

if __name__ == '__main__':
    main()