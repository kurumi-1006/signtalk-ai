import { useState } from 'react';
import { Pressable, SafeAreaView, ScrollView, StyleSheet, Switch, Text, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { router } from 'expo-router';

export default function Settings() {
  const [speech, setSpeech] = useState(true);
  const [vibration, setVibration] = useState(true);
  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView contentContainerStyle={styles.page}>
        <View style={styles.header}>
          <Pressable onPress={() => router.back()} style={styles.iconButton}>
            <Ionicons name="arrow-back" size={20} color="#34433C" />
          </Pressable>
          <Text style={styles.headerTitle}>Settings</Text>
          <View style={styles.headerSpacer} />
        </View>

        <View style={styles.profile}>
          <View style={styles.avatar}><Text style={styles.avatarText}>Q</Text></View>
          <View style={styles.profileCopy}>
            <Text style={styles.profileName}>UNO Q EDGE AI</Text>
            <Text style={styles.profileEmail}>Live recognition on the local network</Text>
          </View>
          <Ionicons name="chevron-forward" size={18} color="#9AA29E" />
        </View>

        <SectionLabel text="RECOGNITION" />
        <View style={styles.card}>
          <SettingRow icon="volume-high-outline" title="Speak results aloud" caption="Read the sentence after recognition is complete" value={speech} onChange={setSpeech} />
          <View style={styles.rule} />
          <SettingRow icon="phone-portrait-outline" title="Haptic feedback" caption="Vibrate gently when a new result is available" value={vibration} onChange={setVibration} />
        </View>

        <SectionLabel text="SYSTEM" />
        <View style={styles.card}>
          <InfoRow icon="language-outline" title="Output language" value="Vietnamese" />
          <View style={styles.rule} />
          <InfoRow icon="hardware-chip-outline" title="Video processing" value="On UNO Q" />
        </View>

        <Text style={styles.version}>SIGNTALK • Live recognition mode</Text>
      </ScrollView>
    </SafeAreaView>
  );
}

function SectionLabel({ text }: { text: string }) {
  return <Text style={styles.sectionLabel}>{text}</Text>;
}

function SettingRow({ icon, title, caption, value, onChange }: { icon: keyof typeof Ionicons.glyphMap; title: string; caption: string; value: boolean; onChange: (value: boolean) => void }) {
  return (
    <View style={styles.row}>
      <View style={styles.rowIcon}><Ionicons name={icon} size={19} color="#4B5A52" /></View>
      <View style={styles.rowCopy}><Text style={styles.rowTitle}>{title}</Text><Text style={styles.rowCaption}>{caption}</Text></View>
      <Switch value={value} onValueChange={onChange} trackColor={{ false: '#D8DDD9', true: '#99B254' }} thumbColor="#FFFFFF" />
    </View>
  );
}

function InfoRow({ icon, title, value }: { icon: keyof typeof Ionicons.glyphMap; title: string; value: string }) {
  return (
    <View style={styles.row}>
      <View style={styles.rowIcon}><Ionicons name={icon} size={19} color="#4B5A52" /></View>
      <Text style={[styles.rowTitle, styles.infoTitle]}>{title}</Text>
      <Text style={styles.infoValue}>{value}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  safeArea: { flex: 1, backgroundColor: '#F4F5F2' },
  page: { alignSelf: 'center', maxWidth: 680, padding: 24, paddingBottom: 42, width: '100%' },
  header: { alignItems: 'center', flexDirection: 'row', justifyContent: 'space-between', marginBottom: 28 },
  iconButton: { alignItems: 'center', backgroundColor: '#FFFFFF', borderColor: '#E0E4E0', borderRadius: 10, borderWidth: 1, height: 40, justifyContent: 'center', width: 40 },
  headerTitle: { color: '#1D2A23', fontSize: 17, fontWeight: '700' }, headerSpacer: { width: 40 },
  profile: { alignItems: 'center', backgroundColor: '#18251F', borderRadius: 14, flexDirection: 'row', gap: 13, padding: 18 },
  avatar: { alignItems: 'center', backgroundColor: '#DFF4A7', borderRadius: 10, height: 44, justifyContent: 'center', width: 44 },
  avatarText: { color: '#17231D', fontSize: 16, fontWeight: '800' }, profileCopy: { flex: 1 },
  profileName: { color: '#FFFFFF', fontSize: 14, fontWeight: '700' }, profileEmail: { color: '#AEB8B2', fontSize: 11, marginTop: 4 },
  sectionLabel: { color: '#78837C', fontSize: 10, fontWeight: '700', letterSpacing: 1.1, marginBottom: 9, marginLeft: 3, marginTop: 26 },
  card: { backgroundColor: '#FFFFFF', borderColor: '#E0E4E0', borderRadius: 12, borderWidth: 1, paddingHorizontal: 15 },
  row: { alignItems: 'center', flexDirection: 'row', gap: 12, minHeight: 70 },
  rowIcon: { alignItems: 'center', backgroundColor: '#F0F2EF', borderRadius: 8, height: 36, justifyContent: 'center', width: 36 },
  rowCopy: { flex: 1 }, rowTitle: { color: '#2A3830', fontSize: 13, fontWeight: '700' }, rowCaption: { color: '#7D8781', fontSize: 11, marginTop: 4 },
  rule: { backgroundColor: '#EAEEEA', height: 1, marginLeft: 48 }, infoTitle: { flex: 1 }, infoValue: { color: '#6F7B74', fontSize: 12, fontWeight: '600' },
  signOut: { alignItems: 'center', borderColor: '#E6C8C4', borderRadius: 10, borderWidth: 1, flexDirection: 'row', gap: 8, justifyContent: 'center', marginTop: 28, paddingVertical: 13 },
  signOutText: { color: '#9A4036', fontSize: 13, fontWeight: '700' }, version: { color: '#9AA29E', fontSize: 10, marginTop: 22, textAlign: 'center' },
});
