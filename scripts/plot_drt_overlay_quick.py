import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV files (skip first 2 header rows, then use row 3 as column names)
drt_4a = pd.read_csv(r'c:\Users\SaiSarbareesh\OneDrive - eleryc inc\Desktop\M_Data\data_extracted\separated_by_current\M1_M1-5psi-07162025\drt\M1_4A_DRT.csv', 
                     skiprows=2)
drt_8a = pd.read_csv(r'c:\Users\SaiSarbareesh\OneDrive - eleryc inc\Desktop\M_Data\data_extracted\separated_by_current\M1_M1-5psi-07162025\drt\M1_8A_DRT.csv', 
                     skiprows=2)

# Create the plot
plt.figure(figsize=(12, 8))
plt.style.use('default')

# Plot 4A data
plt.plot(drt_4a['tau'], drt_4a['gamma'], 'o-', 
         linewidth=2.5, markersize=4,
         color='#e41a1c', label='M1 @ 4A')

# Plot 8A data
plt.plot(drt_8a['tau'], drt_8a['gamma'], 'o-', 
         linewidth=2.5, markersize=4,
         color='#377eb8', label='M1 @ 8A')

# Format the plot
plt.xlabel('τ (s)', fontsize=13)
plt.ylabel('Y (Ω)', fontsize=13)
plt.title('M1 - DRT Overlay (4A vs 8A)', fontsize=15, pad=15)
plt.xscale('log')
plt.grid(True, alpha=0.4, which='both')
plt.legend(fontsize=12, loc='upper right', frameon=True, fancybox=True, shadow=False)
plt.tight_layout()

# Save the plot
output_file = 'M1_4A_vs_8A_DRT_overlay.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
print(f"[+] Plot saved as: {output_file}")

# Show the plot
plt.show()

