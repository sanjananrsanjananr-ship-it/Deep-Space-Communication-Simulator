import math

def link_metrics(distance_km, frequency_hz=2.2e9, tx_dbm=20.0, antenna_gain_db=24.0,
                 system_loss_db=3.0, attenuation_db=2.0, noise_dbm=-100.0, bandwidth_hz=1e6):
    """Educational free-space link estimate. Units: km, Hz, dBm, dB."""
    if distance_km <= 0:
        raise ValueError("distance_km must be positive")
    fspl = 20 * math.log10(distance_km) + 20 * math.log10(frequency_hz) - 147.55
    rx_dbm = tx_dbm + antenna_gain_db - fspl - system_loss_db - attenuation_db
    snr_db = rx_dbm - noise_dbm
    snr_linear = 10 ** (snr_db / 10)
    ber = 0.5 * math.erfc(math.sqrt(max(snr_linear, 0)))
    packet_loss = 1 - (1 - min(max(ber, 0), 0.5)) ** 8000
    packet_loss = min(max(packet_loss, 0), 1)
    return {"distance": distance_km, "delay_s": distance_km / 299792.458, "fspl_db": fspl,
            "rx_dbm": rx_dbm, "snr_db": snr_db, "ber": ber, "packet_loss": packet_loss,
            "throughput_bps": bandwidth_hz * (1 - packet_loss)}

def link_quality(snr_db, packet_loss):
    if snr_db >= 12 and packet_loss < 0.05:
        return "GOOD"
    if snr_db >= 3 and packet_loss < 0.25:
        return "MODERATE"
    return "POOR"
