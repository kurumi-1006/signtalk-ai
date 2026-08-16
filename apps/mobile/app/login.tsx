import { useState } from 'react';
import { ActivityIndicator, Pressable, SafeAreaView, StyleSheet, Text, TextInput, useWindowDimensions, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { router } from 'expo-router';

export default function Login() {
  const { width } = useWindowDimensions();
  const desktop = width >= 800;
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const signIn = async () => {
    try {
      setLoading(true);
      setError('');
      router.replace('/');
    } catch {
      setError('Không thể đăng nhập. Vui lòng kiểm tra lại thông tin.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <View style={[styles.page, desktop && styles.pageDesktop]}>
        <View style={[styles.hero, desktop && styles.heroDesktop]}>
          <View style={styles.logo}><Text style={styles.logoText}>S</Text></View>
          <Text style={styles.brand}>SIGNTALK</Text>
          <Text style={styles.heroTitle}>Giao tiếp không còn khoảng cách.</Text>
          <Text style={styles.heroCopy}>Chuyển ngôn ngữ ký hiệu thành văn bản và giọng nói theo thời gian thực.</Text>
          <View style={styles.featureList}>
            <Feature icon="scan-outline" text="Nhận diện trực tiếp bằng V6.2" />
            <Feature icon="chatbubble-ellipses-outline" text="Hoàn thiện câu theo ngữ cảnh" />
            <Feature icon="volume-high-outline" text="Phát giọng nói tiếng Việt" />
          </View>
        </View>

        <View style={[styles.formPanel, desktop && styles.formPanelDesktop]}>
          <View style={styles.form}>
            <Text style={styles.formKicker}>CHÀO MỪNG TRỞ LẠI</Text>
            <Text style={styles.formTitle}>Đăng nhập</Text>
            <Text style={styles.formSubtitle}>Sử dụng tài khoản SIGNTALK của bạn.</Text>
            <Text style={styles.label}>Email</Text>
            <TextInput value={email} onChangeText={setEmail} autoCapitalize="none" keyboardType="email-address" placeholder="email@example.com" placeholderTextColor="#98A19C" style={styles.input} />
            <Text style={styles.label}>Mật khẩu</Text>
            <TextInput value={password} onChangeText={setPassword} secureTextEntry placeholder="Nhập mật khẩu" placeholderTextColor="#98A19C" style={styles.input} />
            {error ? <View style={styles.error}><Ionicons name="alert-circle-outline" size={17} color="#9A4036" /><Text style={styles.errorText}>{error}</Text></View> : null}
            <Pressable disabled={loading} onPress={signIn} style={[styles.button, loading && styles.buttonDisabled]}>
              {loading ? <ActivityIndicator color="#17231D" /> : <><Text style={styles.buttonText}>Đăng nhập</Text><Ionicons name="arrow-forward" size={17} color="#17231D" /></>}
            </Pressable>
          </View>
          <Text style={styles.footer}>© 2026 SIGNTALK AI</Text>
        </View>
      </View>
    </SafeAreaView>
  );
}

function Feature({ icon, text }: { icon: keyof typeof Ionicons.glyphMap; text: string }) {
  return <View style={styles.feature}><Ionicons name={icon} size={19} color="#DFF4A7" /><Text style={styles.featureText}>{text}</Text></View>;
}

const styles = StyleSheet.create({
  safeArea: { flex: 1, backgroundColor: '#F4F5F2' }, page: { flex: 1, padding: 24 }, pageDesktop: { flexDirection: 'row', padding: 0 },
  hero: { backgroundColor: '#18251F', borderRadius: 16, padding: 28 }, heroDesktop: { borderRadius: 0, flex: 1.05, justifyContent: 'center', padding: 70 },
  logo: { alignItems: 'center', backgroundColor: '#DFF4A7', borderRadius: 10, height: 42, justifyContent: 'center', width: 42 }, logoText: { color: '#17231D', fontSize: 18, fontWeight: '800' },
  brand: { color: '#C7D1CB', fontSize: 10, fontWeight: '800', letterSpacing: 1.7, marginTop: 20 },
  heroTitle: { color: '#FFFFFF', fontSize: 34, fontWeight: '700', letterSpacing: -0.8, lineHeight: 42, marginTop: 13, maxWidth: 480 },
  heroCopy: { color: '#ABB7B0', fontSize: 14, lineHeight: 21, marginTop: 12, maxWidth: 430 },
  featureList: { gap: 13, marginTop: 34 }, feature: { alignItems: 'center', flexDirection: 'row', gap: 11 }, featureText: { color: '#D4DDD7', fontSize: 12 },
  formPanel: { flex: 1, justifyContent: 'center', paddingVertical: 38 }, formPanelDesktop: { padding: 64 },
  form: { alignSelf: 'center', maxWidth: 410, width: '100%' }, formKicker: { color: '#7A867F', fontSize: 10, fontWeight: '700', letterSpacing: 1.1 },
  formTitle: { color: '#1D2A23', fontSize: 29, fontWeight: '700', marginTop: 8 }, formSubtitle: { color: '#78837C', fontSize: 13, marginBottom: 28, marginTop: 6 },
  label: { color: '#4D5A53', fontSize: 11, fontWeight: '700', marginBottom: 7, marginTop: 13 },
  input: { backgroundColor: '#FFFFFF', borderColor: '#DCE2DD', borderRadius: 10, borderWidth: 1, color: '#24322B', fontSize: 14, height: 50, paddingHorizontal: 14 },
  error: { alignItems: 'center', backgroundColor: '#FFF3F1', borderRadius: 8, flexDirection: 'row', gap: 7, marginTop: 14, padding: 10 }, errorText: { color: '#8E3D34', flex: 1, fontSize: 11 },
  button: { alignItems: 'center', backgroundColor: '#DFF4A7', borderRadius: 10, flexDirection: 'row', gap: 8, justifyContent: 'center', marginTop: 20, minHeight: 51 },
  buttonDisabled: { opacity: 0.6 }, buttonText: { color: '#17231D', fontSize: 13, fontWeight: '700' },
  footer: { color: '#9AA29E', fontSize: 10, marginTop: 30, textAlign: 'center' },
});
