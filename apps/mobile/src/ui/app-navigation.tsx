import { Pressable, StyleSheet, Text, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { router, usePathname } from 'expo-router';

const items = [
  { href: '/', icon: 'scan-outline', label: 'Phiên dịch' },
  { href: '/history', icon: 'time-outline', label: 'Lịch sử' },
] as const;

export function AppNavigation({ variant = 'bottom' }: { variant?: 'bottom' | 'sidebar' }) {
  const pathname = usePathname();
  const sidebar = variant === 'sidebar';
  return (
    <View style={sidebar ? styles.sidebar : styles.bottom}>
      {sidebar ? (
        <View style={styles.brandMark}>
          <View style={styles.brandIcon}><Text style={styles.brandLetter}>S</Text></View>
          <Text style={styles.brandName}>SIGNTALK</Text>
        </View>
      ) : null}
      <View style={sidebar ? styles.sidebarItems : styles.bottomItems}>
        {items.map((item) => {
          const active = pathname === item.href;
          return (
            <Pressable
              key={item.href}
              accessibilityRole="button"
              onPress={() => router.replace(item.href as never)}
              style={[
                sidebar ? styles.sidebarItem : styles.bottomItem,
                active && (sidebar ? styles.sidebarItemActive : styles.bottomItemActive),
              ]}
            >
              <Ionicons
                name={item.icon}
                size={sidebar ? 19 : 20}
                color={active ? '#20301F' : '#7B8580'}
              />
              <Text style={[styles.label, active && styles.activeLabel]}>{item.label}</Text>
            </Pressable>
          );
        })}
      </View>
      {sidebar ? (
        <Pressable onPress={() => router.push('/settings')} style={styles.sidebarSettings}>
          <Ionicons name="settings-outline" size={19} color="#7B8580" />
          <Text style={styles.label}>Cài đặt</Text>
        </Pressable>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  bottom: { backgroundColor: '#FFFFFF', borderColor: '#E0E4E0', borderRadius: 14, borderWidth: 1, marginTop: 2, padding: 5 },
  bottomItems: { flexDirection: 'row', justifyContent: 'space-around' },
  bottomItem: { alignItems: 'center', borderRadius: 9, flex: 1, gap: 4, paddingVertical: 8 },
  bottomItemActive: { backgroundColor: '#EEF5D9' },
  sidebar: { backgroundColor: '#FFFFFF', borderRightColor: '#E1E5E1', borderRightWidth: 1, minHeight: '100%', paddingHorizontal: 14, paddingVertical: 24, width: 214 },
  brandMark: { alignItems: 'center', flexDirection: 'row', gap: 10, paddingHorizontal: 9 },
  brandIcon: { alignItems: 'center', backgroundColor: '#1D2B24', borderRadius: 8, height: 32, justifyContent: 'center', width: 32 },
  brandLetter: { color: '#E6F7B4', fontSize: 15, fontWeight: '800' },
  brandName: { color: '#1F2D26', fontSize: 12, fontWeight: '800', letterSpacing: 0.9 },
  sidebarItems: { gap: 5, marginTop: 38 },
  sidebarItem: { alignItems: 'center', borderRadius: 9, flexDirection: 'row', gap: 11, paddingHorizontal: 12, paddingVertical: 11 },
  sidebarItemActive: { backgroundColor: '#EEF5D9' },
  sidebarSettings: { alignItems: 'center', borderTopColor: '#E8EBE8', borderTopWidth: 1, flexDirection: 'row', gap: 11, marginTop: 'auto', paddingHorizontal: 12, paddingTop: 18 },
  label: { color: '#737E77', fontSize: 12, fontWeight: '600' },
  activeLabel: { color: '#20301F', fontWeight: '700' },
});
