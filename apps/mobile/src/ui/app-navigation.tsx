import { Pressable, StyleSheet, Text, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { router, usePathname } from 'expo-router';

const items = [
  { href: '/', icon: 'scan-outline', label: 'Translate' },
  { href: '/history', icon: 'time-outline', label: 'History' },
] as const;

export function AppNavigation({ variant = 'bottom' }: { variant?: 'bottom' | 'sidebar' }) {
  const pathname = usePathname();
  const sidebar = variant === 'sidebar';
  return (
    <View style={sidebar ? styles.sidebar : styles.bottom}>
      {sidebar ? (
        <View style={styles.brandMark}>
          <View style={styles.brandIcon}><Text style={styles.brandLetter}>S</Text></View>
          <Text style={styles.brandName}>SIGNTALK AI</Text>
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
    </View>
  );
}

const styles = StyleSheet.create({
  bottom: { backgroundColor: '#FFFFFF', borderColor: '#DFE8DE', borderRadius: 18, borderWidth: 1, marginTop: 2, padding: 6 },
  bottomItems: { flexDirection: 'row', justifyContent: 'space-around' },
  bottomItem: { alignItems: 'center', borderRadius: 12, flex: 1, gap: 4, paddingVertical: 9 },
  bottomItemActive: { backgroundColor: '#EEF6DC' },
  sidebar: { backgroundColor: '#FFFFFF', borderRightColor: '#E1E8E0', borderRightWidth: 1, minHeight: '100%', paddingHorizontal: 16, paddingVertical: 26, width: 224 },
  brandMark: { alignItems: 'center', flexDirection: 'row', gap: 10, paddingHorizontal: 9 },
  brandIcon: { alignItems: 'center', backgroundColor: '#173125', borderRadius: 10, height: 36, justifyContent: 'center', width: 36 },
  brandLetter: { color: '#E6F7B4', fontSize: 16, fontWeight: '800' },
  brandName: { color: '#1F2D26', fontSize: 13, fontWeight: '800', letterSpacing: 1 },
  sidebarItems: { gap: 7, marginTop: 42 },
  sidebarItem: { alignItems: 'center', borderRadius: 12, flexDirection: 'row', gap: 11, paddingHorizontal: 13, paddingVertical: 13 },
  sidebarItemActive: { backgroundColor: '#EEF6DC' },
  label: { color: '#737E77', fontSize: 12, fontWeight: '600' },
  activeLabel: { color: '#20301F', fontWeight: '700' },
});
