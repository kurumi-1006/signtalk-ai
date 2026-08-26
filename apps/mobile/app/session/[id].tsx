import { useMemo } from 'react';
import { Alert, Pressable, SafeAreaView, ScrollView, StyleSheet, Text, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { router, useLocalSearchParams } from 'expo-router';
import { useRecognitionStore } from '../../src/store/recognition';

export default function Session() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const { events } = useRecognitionStore();
  const started = useMemo(
    () => new Intl.DateTimeFormat('en-US', { hour: '2-digit', minute: '2-digit' }).format(new Date()),
    [],
  );
  const average = events.length
    ? Math.round(events.reduce((sum, item) => sum + item.payload.confidence, 0) / events.length * 100)
    : 0;

  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView contentContainerStyle={styles.page}>
        <View style={styles.header}>
          <Pressable onPress={() => router.back()} style={styles.iconButton}><Ionicons name="arrow-back" size={20} color="#34433C" /></Pressable>
          <Text style={styles.headerTitle}>Session details</Text>
          <Pressable onPress={() => Alert.alert('Export record', 'The record will be exported when data is available.')} style={styles.iconButton}>
            <Ionicons name="download-outline" size={20} color="#34433C" />
          </Pressable>
        </View>

        <View style={styles.hero}>
          <View style={styles.liveRow}><View style={styles.liveDot} /><Text style={styles.liveLabel}>{id === 'live' ? 'LIVE' : 'SAVED'}</Text></View>
          <Text style={styles.heroTitle}>Recognition session</Text>
          <Text style={styles.heroMeta}>Started at {started} • {events.length} results</Text>
        </View>

        <View style={styles.stats}>
          <Stat icon="chatbubble-outline" value={String(events.length)} label="Results" />
          <View style={styles.ruleVertical} />
          <Stat icon="analytics-outline" value={events.length ? `${average}%` : '—'} label="Avg. confidence" />
        </View>

        <Text style={styles.sectionTitle}>Session record</Text>
        <View style={styles.list}>
          {events.length ? events.map((event, index) => (
            <View key={event.eventId} style={[styles.row, index > 0 && styles.rowBorder]}>
              <View style={styles.rowIndex}><Text style={styles.rowIndexText}>{index + 1}</Text></View>
              <View style={styles.rowCopy}>
                <Text style={styles.rowText}>{event.payload.text}</Text>
                <Text style={styles.rowMeta}>{event.payload.label} • {Math.round(event.payload.confidence * 100)}% confidence</Text>
              </View>
              <Ionicons name="checkmark-circle" size={19} color="#7E9845" />
            </View>
          )) : (
            <View style={styles.empty}>
              <Ionicons name="document-text-outline" size={27} color="#77837C" />
              <Text style={styles.emptyText}>No results in this session yet.</Text>
            </View>
          )}
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

function Stat({ icon, value, label }: { icon: keyof typeof Ionicons.glyphMap; value: string; label: string }) {
  return (
    <View style={styles.stat}>
      <Ionicons name={icon} size={19} color="#73804E" />
      <Text style={styles.statValue}>{value}</Text>
      <Text style={styles.statLabel}>{label}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  safeArea: { flex: 1, backgroundColor: '#F6F8F5' },
  page: { alignSelf: 'center', gap: 18, maxWidth: 780, padding: 28, paddingBottom: 42, width: '100%' },
  header: { alignItems: 'center', flexDirection: 'row', justifyContent: 'space-between' },
  iconButton: { alignItems: 'center', backgroundColor: '#FFFFFF', borderColor: '#DFE8DE', borderRadius: 12, borderWidth: 1, height: 44, justifyContent: 'center', width: 44 },
  headerTitle: { color: '#14221B', fontSize: 18, fontWeight: '800' },
  hero: { backgroundColor: '#173125', borderRadius: 20, padding: 22 },
  liveRow: { alignItems: 'center', flexDirection: 'row', gap: 7 }, liveDot: { backgroundColor: '#F16A5B', borderRadius: 4, height: 7, width: 7 },
  liveLabel: { color: '#C2D39D', fontSize: 9, fontWeight: '700', letterSpacing: 1 }, heroTitle: { color: '#FFFFFF', fontSize: 23, fontWeight: '700', marginTop: 14 },
  heroMeta: { color: '#AEB8B2', fontSize: 12, marginTop: 6 },
  stats: { alignItems: 'center', backgroundColor: '#FFFFFF', borderColor: '#DFE8DE', borderRadius: 16, borderWidth: 1, flexDirection: 'row', paddingVertical: 19 },
  stat: { alignItems: 'center', flex: 1, gap: 4 }, statValue: { color: '#26352E', fontSize: 20, fontWeight: '700' }, statLabel: { color: '#808A84', fontSize: 10 },
  ruleVertical: { backgroundColor: '#E8ECE8', height: 40, width: 1 }, sectionTitle: { color: '#28362F', fontSize: 15, fontWeight: '700', marginTop: 4 },
  list: { backgroundColor: '#FFFFFF', borderColor: '#DFE8DE', borderRadius: 18, borderWidth: 1, overflow: 'hidden' },
  row: { alignItems: 'center', flexDirection: 'row', gap: 12, minHeight: 72, paddingHorizontal: 15 }, rowBorder: { borderTopColor: '#E9ECE9', borderTopWidth: 1 },
  rowIndex: { alignItems: 'center', backgroundColor: '#F0F2EF', borderRadius: 8, height: 34, justifyContent: 'center', width: 34 }, rowIndexText: { color: '#647169', fontSize: 11, fontWeight: '700' },
  rowCopy: { flex: 1, gap: 4 }, rowText: { color: '#29372F', fontSize: 13, fontWeight: '700' }, rowMeta: { color: '#818B85', fontSize: 10 },
  empty: { alignItems: 'center', gap: 9, padding: 38 }, emptyText: { color: '#7B867F', fontSize: 12 },
});
