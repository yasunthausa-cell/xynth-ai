const {
    default: makeWASocket,
    useMultiFileAuthState,
    DisconnectReason,
    fetchLatestBaileysVersion,
} = require('@whiskeysockets/baileys');
const qrcode = require('qrcode-terminal');
const axios = require('axios');
const pino = require('pino');

const API_URL = process.env.AGENT_API_URL || 'http://localhost:8000/chat';
const AUTH_DIR = '.wwebjs_auth';

async function start() {
    const { state, saveCreds } = await useMultiFileAuthState(AUTH_DIR);
    const { version } = await fetchLatestBaileysVersion();

    const sock = makeWASocket({
        version,
        auth: state,
        logger: pino({ level: 'silent' }),
        printQRInTerminal: false,
        browser: ['Xynth AI', 'Chrome', '120.0'],
    });

    sock.ev.on('creds.update', saveCreds);

    sock.ev.on('connection.update', (update) => {
        const { connection, lastDisconnect, qr } = update;
        if (qr) {
            console.log('\n📱 Open WhatsApp on your phone → Settings → Linked Devices → Link a Device, then scan:\n');
            qrcode.generate(qr, { small: true });
        }
        if (connection === 'close') {
            const code = lastDisconnect?.error?.output?.statusCode;
            const shouldReconnect = code !== DisconnectReason.loggedOut;
            console.log(`Connection closed (code ${code}). Reconnecting: ${shouldReconnect}`);
            if (shouldReconnect) setTimeout(start, 2000);
        } else if (connection === 'open') {
            console.log('✅ WhatsApp connected! Send the bot a message from any chat.');
        }
    });

    sock.ev.on('messages.upsert', async ({ messages, type }) => {
        if (type !== 'notify') return;
        for (const m of messages) {
            if (m.key.fromMe) continue;
            const remoteJid = m.key.remoteJid;
            if (!remoteJid || remoteJid === 'status@broadcast') continue;
            // Skip groups — only respond in 1:1 chats
            if (remoteJid.endsWith('@g.us')) continue;

            const text =
                m.message?.conversation ||
                m.message?.extendedTextMessage?.text ||
                '';
            if (!text.trim()) continue;

            console.log(`💬 ${remoteJid}: ${text.slice(0, 80)}`);

            try {
                await sock.sendPresenceUpdate('composing', remoteJid);
                const resp = await axios.post(
                    API_URL,
                    {
                        session_id: `wa-${remoteJid}`,
                        message: text,
                    },
                    { timeout: 180000 }
                );
                const reply = resp.data?.response || '(no response)';
                await sock.sendMessage(remoteJid, { text: reply });
                console.log(`✅ Replied to ${remoteJid}`);
            } catch (err) {
                console.error('Error:', err.message);
                try {
                    await sock.sendMessage(remoteJid, {
                        text: `❌ Sorry, I hit an error: ${err.message}`,
                    });
                } catch (_) {}
            }
        }
    });
}

start().catch((e) => {
    console.error('Fatal:', e);
    process.exit(1);
});
