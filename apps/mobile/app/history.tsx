import { useMemo, useState } from 'react';
import { Pressable, SafeAreaView, ScrollView, StyleSheet, Text, useWindowDimensions, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { router } from 'expo-router';
import { useRecognitionStore } from '../src/store/recognition';
import { AppNavigation } from '../src/ui/app-navigation';

export default function History() {
  const { width } = useWindowDimensions();
  const desktop = width >= 900;
  const { events } = useRecognitionStore();
  const [filter, setFilter] = useState<'today' | 'all'>('today');
  const rows = useMemo(() => (filter === 'today' ? events : events), [events, filter]);

  return (
    <SafeAreaView style={styles.safeArea}>
      <View style={styles.shell}>
        {desktop ? <AppNavigation variant="sidebar" /> : null}
        <ScrollView contentContainerStyle={styles.page} showsVerticalScrollIndicator={false}>
          <View style={styles.header}>
            <View>
              <Text style={styles.kicker}>ACTIVITY</Text>
              <Text style={styles.title}>Recognition history</Text>
              <Text style={styles.subtitle}>Review sentences and model confidence.</Text>
            </View>
            <Pressable onPress={() => router.push('/settings')} style={styles.iconButton}>
              <Ionicons name="settings-outline" size={20} color="#34433C" />
            </Pressable>
          </View>

          <Pressable onPress={() => router.push('/session/live')} style={styles.sessionCard}>
            <View style={styles.sessionIcon}><Ionicons name="radio-outline" size={21} color="#DFF4A7" /></View>
            <View style={styles.sessionCopy}>
              <Text style={styles.sessionKicker}>CURRENT SESSION</Text>
              <Text style={styles.sessionTitle}>Live recognition</Text>
              <Text style={styles.sessionMeta}>{events.length} results recorded</Text>
            </View>
            <Ionicons name="arrow-forward" size={20} color="#FFFFFF" />
          </Pressable>

          <View style={styles.toolbar}>
            <View style={styles.segment}>
              <Pressable onPress={() => setFilter('today')} style={[styles.segmentItem, filter === 'today' && styles.segmentActive]}>
                <Text style={[styles.segmentText, filter === 'today' && styles.segmentTextActive]}>Today</Text>
              </Pressable>
              <Pressable onPress={() => setFilter('all')} style={[styles.segmentItem, filter === 'all' && styles.segmentActive]}>
                <Text style={[styles.segmentText, filter === 'all' && styles.segmentTextActive]}>All</Text>
              </Pressable>
            </View>
            <Text style={styles.count}>{rows.length} results</Text>
          </View>

          <View style={styles.list}>
            {rows.length ? rows.map((event, index) => (
              <View key={event.eventId} style={[styles.row, index > 0 && styles.rowBorder]}>
                <View style={styles.rowIcon}><Ionicons name="hand-left-outline" size={19} color="#435149" /></View>
                <View style={styles.rowCopy}>
                  <Text style={styles.rowText}>{event.payload.text}</Text>
                  <Text style={styles.rowMeta}>{event.payload.label} • {Math.round(event.payload.confidence * 100)}% confidence</Text>
                </View>
                <Pressable style={styles.playButton}><Ionicons name="play" size={14} color="#36443D" /></Pressable>
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
  safeArea: { flex: 1, backgroundColor: '#F4F5F2' }, shell: { flex: 1, flexDirection: 'row' },
  page: { alignSelf: 'center', gap: 22, maxWidth: 900, padding: 24, paddingBottom: 38, width: '100%' },
  header: { alignItems: 'flex-start', flexDirection: 'row', justifyContent: 'space-between' },
  kicker: { color: '#737E77', fontSize: 10, fontWeight: '700', letterSpacing: 1.2 },
  title: { color: '#17231D', fontSize: 28, fontWeight: '700', letterSpacing: -0.6, marginTop: 6 },
  subtitle: { color: '#707A74', fontSize: 14, marginTop: 6 },
  iconButton: { alignItems: 'center', backgroundColor: '#FFFFFF', borderColor: '#E0E4E0', borderRadius: 10, borderWidth: 1, height: 40, justifyContent: 'center', width: 40 },
  sessionCard: { alignItems: 'center', backgroundColor: '#18251F', borderRadius: 14, flexDirection: 'row', gap: 14, padding: 18 },
  sessionIcon: { alignItems: 'center', borderColor: '#415048', borderRadius: 10, borderWidth: 1, height: 44, justifyContent: 'center', width: 44 },
  sessionCopy: { flex: 1, gap: 3 }, sessionKicker: { color: '#B7C99A', fontSize: 9, fontWeight: '700', letterSpacing: 1 },
  sessionTitle: { color: '#FFFFFF', fontSize: 15, fontWeight: '700' }, sessionMeta: { color: '#AEB8B2', fontSize: 11 },
  toolbar: { alignItems: 'center', flexDirection: 'row', justifyContent: 'space-between' },
  segment: { backgroundColor: '#E7EAE6', borderRadius: 9, flexDirection: 'row', padding: 3 },
  segmentItem: { borderRadius: 7, paddingHorizontal: 16, paddingVertical: 8 }, segmentActive: { backgroundColor: '#FFFFFF' },
  segmentText: { color: '#79847D', fontSize: 12, fontWeight: '600' }, segmentTextActive: { color: '#26352E' },
  count: { color: '#7C8780', fontSize: 12 },
  list: { backgroundColor: '#FFFFFF', borderColor: '#E0E4E0', borderRadius: 13, borderWidth: 1, overflow: 'hidden' },
  row: { alignItems: 'center', flexDirection: 'row', gap: 12, minHeight: 76, paddingHorizontal: 16 },
  rowBorder: { borderTopColor: '#E9ECE9', borderTopWidth: 1 },
  rowIcon: { alignItems: 'center', backgroundColor: '#F0F2EF', borderRadius: 9, height: 40, justifyContent: 'center', width: 40 },
  rowCopy: { flex: 1, gap: 4 }, rowText: { color: '#29372F', fontSize: 14, fontWeight: '700' }, rowMeta: { color: '#818B85', fontSize: 11 },
  playButton: { alignItems: 'center', borderColor: '#DEE3DE', borderRadius: 8, borderWidth: 1, height: 34, justifyContent: 'center', width: 34 },
  empty: { alignItems: 'center', padding: 44 }, emptyIcon: { alignItems: 'center', backgroundColor: '#EFF1EE', borderRadius: 12, height: 50, justifyContent: 'center', width: 50 },
  emptyTitle: { color: '#2A3830', fontSize: 16, fontWeight: '700', marginTop: 15 }, emptyText: { color: '#7B867F', fontSize: 13, lineHeight: 19, marginTop: 6, textAlign: 'center' },
  emptyButton: { alignItems: 'center', backgroundColor: '#DFF4A7', borderRadius: 9, flexDirection: 'row', gap: 8, marginTop: 18, paddingHorizontal: 16, paddingVertical: 11 },
  emptyButtonText: { color: '#142019', fontSize: 12, fontWeight: '700' },
});
