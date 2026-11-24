f1_comparison = Lat_Lon_f1.join(Sea_Wave_f1['Sea/Wave f1'])

print(f1_comparison)

ax = f1_comparison.plot(figsize=(12, 12))
ax.set(xlabel='Num of Neighbors', ylabel='F1 Score')
ax.set_xticks(range(1, max_k, 2));
plt.ylim((0, 1))
plt.title('KNN F1 Score')

plt.show()