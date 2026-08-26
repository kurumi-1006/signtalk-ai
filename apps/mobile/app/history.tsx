import { Pressable, SafeAreaView, ScrollView, StyleSheet, Text, useWindowDimensions, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { router } from 'expo-router';
import { useRecognitionStore } from '../src/store/recognition';
import { AppNavigation } from '../src/ui/app-navigation';

export default function History() {
  const { width } = useWindowDimensions();
  const desktop = width >= 980;
  const compact = width < 640;
  const { events } = useRecognitionStore();

  return (
    <SafeAreaView style={styles.safeArea}>
      <View style={styles.shell}>
        {desktop ? <AppNavigation variant="sidebar" /> : null}
        <ScrollView contentContainerStyle={[styles.page, compact && styles.pageCompact]} showsVerticalScrollIndicator={false}>
          <View style={styles.header}>
            <View>
              <Text style={styles.title}>Recognition history</Text>
            </View>
          </View>

          <View style={styles.toolbar}>
            <Text style={styles.count}>{events.length} {events.length === 1 ? 'result' : 'results'}</Text>
          </View>

          <View style={styles.list}>
            {events.length ? events.map((event, index) => (
              <View key={event.eventId} style={[styles.row, index > 0 && styles.rowBorder]}>
                <View style={styles.rowIcon}><Ionicons name="hand-left-outline" size={19} color="#435149" /></View>
                <View style={styles.rowCopy}>
                  <Text style={styles.rowText}>{event.payload.text}</Text>
                  <Text style={styles.rowMeta}>{event.payload.label} • {Math.round(event.payload.confidence * 100)}% confidence</Text>
                </View>
              </View>
            )) : (
              <View style={styles.empty}>
                <View style={styles.emptyIcon}><Ionicons name="time-outline" size={26} color="#718078" /></View>
                <Text style={styles.emptyTitle}>No history yet</Text>
                <Text style={styles.emptyText}>Start a recognition session to see results here.</Text>
                <Pressable onPress={() => router.replace('/')} style={styles.emptyButton}>
                  <Text style={styles.emptyButtonText}>Open camera</Text>
                  <Ionicons name="arrow-forward" size={16} color="#142019" />
                </Pressable>
              </View>
            )}
          </View>
          {!desktop ? <AppNavigation /> : null}
        </ScrollView>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: { flex: 1, backgroundColor: '#F6F8F5' }, shell: { flex: 1, flexDirection: 'row' },
  page: { alignSelf: 'center', gap: 16, maxWidth: 820, padding: 28, paddingBottom: 42, width: '100%' },
  pageCompact: { gap: 14, padding: 16, paddingBottom: 24 },
  header: { alignItems: 'flex-start', flexDirection: 'row', justifyContent: 'space-between' },
  title: { color: '#14221B', fontSize: 32, fontWeight: '800', letterSpacing: -1, marginTop: 2 },
  iconButton: { alignItems: 'center', backgroundColor: '#FFFFFF', borderColor: '#E0E4E0', borderRadius: 10, borderWidth: 1, height: 40, justifyContent: 'center', width: 40 },
  toolbar: { alignItems: 'center', flexDirection: 'row', justifyContent: 'flex-end', minHeight: 28 },
  count: { color: '#728078', fontSize: 12, fontWeight: '700' },
  list: { backgroundColor: '#FFFFFF', borderColor: '#DFE8DE', borderRadius: 20, borderWidth: 1, overflow: 'hidden' },
  row: { alignItems: 'center', flexDirection: 'row', gap: 13, minHeight: 82, paddingHorizontal: 18 },
  rowBorder: { borderTopColor: '#EAF0E8', borderTopWidth: 1 },
  rowIcon: { alignItems: 'center', backgroundColor: '#EFF5E8', borderRadius: 12, height: 44, justifyContent: 'center', width: 44 },
  rowCopy: { flex: 1, gap: 5 }, rowText: { color: '#24352B', fontSize: 15, fontWeight: '800' }, rowMeta: { color: '#7A887F', fontSize: 12 },
  empty: { alignItems: 'center', padding: 56 }, emptyIcon: { alignItems: 'center', backgroundColor: '#EFF5E8', borderRadius: 16, height: 58, justifyContent: 'center', width: 58 },
  emptyTitle: { color: '#2A3830', fontSize: 18, fontWeight: '800', marginTop: 17 }, emptyText: { color: '#7B867F', fontSize: 13, lineHeight: 20, marginTop: 7, textAlign: 'center' },
  emptyButton: { alignItems: 'center', backgroundColor: '#DDF5A6', borderRadius: 12, flexDirection: 'row', gap: 8, marginTop: 20, paddingHorizontal: 18, paddingVertical: 13 },
  emptyButtonText: { color: '#142019', fontSize: 13, fontWeight: '800' },
});
