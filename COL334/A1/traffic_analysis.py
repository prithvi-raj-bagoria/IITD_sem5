import argparse
from collections import defaultdict
import statistics
import matplotlib.pyplot as plt
from scapy.all import rdpcap
from scapy.layers.inet import IP, TCP
from scapy.layers.inet6 import IPv6


# ---------------------------- helpers ---------------------------- #

def ip_layer(pkt):
    """Return IPv6 or IPv4 layer, else None."""
    if IPv6 in pkt:
        return pkt[IPv6]
    if IP in pkt:
        return pkt[IP]
    return None


def bidir_flow(packets, client, server):
    """TCP packets strictly between client and server (both directions)."""
    out = []
    for p in packets:
        if TCP not in p:
            continue
        ip = ip_layer(p)
        if not ip:
            continue
        s, d = ip.src, ip.dst
        if (s == client and d == server) or (s == server and d == client):
            out.append(p)
    return out


# ------------------------ throughput (1 s bins) ------------------------ #

def throughput_series(flow, client, server, direction, t0):
    """Return (seconds_from_start, Mbps) in 1-second bins.
    direction: 'down' = server→client, 'up' = client→server."""
    if not flow:
        return [], []

    bins = defaultdict(int)  # bytes per second
    for p in flow:
        if TCP not in p:
            continue
        ip = ip_layer(p)
        if not ip:
            continue
        s, d = ip.src, ip.dst
        sec = int(p.time) - t0
        if direction == 'down' and s == server and d == client:
            bins[sec] += len(p[TCP].payload)
        elif direction == 'up' and s == client and d == server:
            bins[sec] += len(p[TCP].payload)

    if not bins:
        return [], []

    xmax = max(bins)
    xs = list(range(0, xmax + 1))
    ys_kbps = [(bins.get(i, 0) * 8) / 1000.0 for i in xs]  # kbps
    return xs, ys_kbps


# ------------------------------ RTT (SYN↔SYN+ACK) ------------------------------ #

def handshake_rtts(flow, client, server):
    """RTTs for TCP handshakes: client SYN → server SYN+ACK."""
    syn_time = {}  # (sport,dport) -> time
    rtts = []
    for p in flow:
        if TCP not in p:
            continue
        ip = ip_layer(p)
        if not ip:
            continue
        tcp = p[TCP]
        s, d = ip.src, ip.dst

        # SYN from client (no ACK bit)
        if s == client and d == server and (tcp.flags & 0x02) and not (tcp.flags & 0x10):
            syn_time[(tcp.sport, tcp.dport)] = p.time
        # SYN+ACK from server
        elif s == server and d == client and (tcp.flags & 0x12) == 0x12:
            key = (tcp.dport, tcp.sport)  # reverse
            if key in syn_time:
                rtts.append(p.time - syn_time.pop(key))
    return rtts


# ------------------------------ plotting ------------------------------ #

def line_plot(x, y, xlabel, ylabel, title, outpng):
    plt.figure(figsize=(9, 5))
    plt.plot(x, y, marker='o', linestyle='-', linewidth=2, markersize=4)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.35)
    if x:
        plt.xlim(0, max(x) + 1)
    plt.tight_layout()
    plt.savefig(outpng, dpi=180)
    plt.close()
    print(f"Saved {outpng}")


# ------------------------------- main ------------------------------- #

def main():
    ap = argparse.ArgumentParser(description='Throughput/RTT from PCAP (IPv4/IPv6).')
    ap.add_argument('--pcap', required=True, help='Path to http.pcap / https.pcap')
    ap.add_argument('--client', required=True, help='Client IP (v4/v6)')
    ap.add_argument('--server', required=True, help='Server IP (v4/v6)')
    ap.add_argument('--throughput', action='store_true', help='Compute throughput')
    ap.add_argument('--down', action='store_true', help='Download throughput')
    ap.add_argument('--up', action='store_true', help='Upload throughput')
    ap.add_argument('--rtt', action='store_true', help='Plot handshake RTTs')
    ap.add_argument('--count-conns', action='store_true', help='Print TCP connection count')
    args = ap.parse_args()

    pcap_path = args.pcap if args.pcap.endswith('.pcap') else f"{args.pcap}.pcap"
    packets = rdpcap(pcap_path)
    flow = bidir_flow(packets, args.client, args.server)

    if not flow:
        print('No TCP packets found for given client/server.')
        return

    t0 = int(min(p.time for p in packets))  # seconds since capture start

    if args.count_conns:
        syns = set()
        for p in flow:
            if TCP in p:
                ip = ip_layer(p)
                tcp = p[TCP]
                if ip and ip.src == args.client and ip.dst == args.server and (tcp.flags & 0x02) and not (tcp.flags & 0x10):
                    syns.add((ip.src, tcp.sport, ip.dst, tcp.dport))
        print(f"TCP connections (client→server SYNs): {len(syns)}")

    if args.throughput:
        if args.down:
            x, y = throughput_series(flow, args.client, args.server, 'down', t0)
            if x:
                line_plot(x, y, 'Time since capture start (s)', 'Download throughput (kbps)', 'Download Throughput', 'down_throughput.png')
            else:
                print('No matching packets for download throughput.')
        if args.up:
            x, y = throughput_series(flow, args.client, args.server, 'up', t0)
            if x:
                line_plot(x, y, 'Time since capture start (s)', 'Upload throughput (kbps)', 'Upload Throughput', 'up_throughput.png')
            else:
                print('No matching packets for upload throughput.')

    if args.rtt:
        rtts = handshake_rtts(flow, args.client, args.server)
        if rtts:
            # Collect SYN packet times to plot RTTs against capture time
            syn_times = []
            for p in flow:
                if TCP in p and ip_layer(p) and ip_layer(p).src == args.client and (p[TCP].flags & 0x02) and not (p[TCP].flags & 0x10):
                    syn_times.append(p.time - t0)  # Time since capture start
            
            # Use syn_times for x-axis if we have them, otherwise just use indexes
            x_values = syn_times[:len(rtts)] if len(syn_times) >= len(rtts) else [t - t0 for t in syn_times]
            
            if not x_values or len(x_values) != len(rtts):
                # Fallback to indexes if times don't match up
                x_values = list(range(len(rtts)))
                line_plot(x_values, rtts, 'Handshake index', 'RTT (s)', 'TCP Handshake RTTs', 'rtt.png')
            else:
                line_plot(x_values, rtts, 'Time since capture start (s)', 'RTT (s)', 'TCP Handshake RTTs', 'rtt.png')
                
            print(f"RTT stats — count: {len(rtts)}, mean: {statistics.mean(rtts):.4f}s, median: {statistics.median(rtts):.4f}s")
        else:
            print('No SYN/SYN+ACK pairs found for RTT.')


if __name__ == '__main__':
    main()
