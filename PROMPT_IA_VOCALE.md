# Prompt : construire une IA vocale à personnalité (Claude + Fish Audio), ultra-réactive, sur PC Windows

> Copie tout ce qui suit dans ton agent de code local (Claude Code, par exemple), à la racine d'un dossier vide.

---

## Rôle et objectif

Tu es un développeur Python senior. Tu vas construire avec moi, sur mon PC Windows, une **IA vocale avec une personnalité** à qui je parle au micro en mains libres. Elle me répond à voix haute, avec des émotions, grâce à l'API **Fish Audio**.

**La priorité absolue, c'est la latence.** Objectif : **1,2 à 2 s maximum** entre la fin de ma phrase et sa première syllabe. Chaque choix technique doit être justifié par rapport à ce délai.

Avant d'écrire du code, pose-moi ces questions en une seule fois :
1. Le nom et la personnalité de l'IA (caractère, façon de parler, tutoiement, humour, limites).
2. Mon ID de voix Fish Audio (`reference_id`) et le modèle Fish à utiliser (S1 ou S2-Pro).
3. Si j'utilise un casque (c'est l'hypothèse par défaut) ou des haut-parleurs.

## Mon environnement

- Windows 10/11, Python 3.11+.
- Carte NVIDIA RTX récente avec CUDA, donc reconnaissance vocale en local sur le GPU.
- Langue de conversation : **français**.
- Clés API : `ANTHROPIC_API_KEY` et `FISH_API_KEY` dans un fichier `.env`. Ne les écris jamais en dur et ne les commite jamais (`.env` va dans `.gitignore`).

## Architecture imposée : tout en flux, rien n'attend

```
Micro ─▶ VAD (Silero) ─▶ STT (faster-whisper, GPU) ─▶ LLM (Claude, streaming)
                                                        │ tokens
                                                        ▼
                                                    Chunker (phrases + émotions)
                                                        │ phrase par phrase
                                                        ▼
                                Fish Audio TTS (WebSocket, connexion maintenue)
                                                        │ PCM en flux
                                                        ▼
                                                 Lecture audio (coupable)
                         Bus d'événements (WebSocket local) ─▶ futur avatar
```

Règle d'or : **une étape n'attend jamais qu'une autre ait fini**. La première phrase de Claude part vers Fish pendant que Claude écrit encore la suite. L'audio se joue dès que les premiers octets PCM arrivent. Le tout est orchestré avec `asyncio`, et les traitements bloquants (Whisper, audio) tournent dans des threads ou exécuteurs dédiés.

## Modules à créer

```
voice_ai/
  main.py        orchestrateur asyncio, boucle de conversation, gestion des interruptions
  config.py      chargement de config.toml + .env, validé (pydantic ou dataclasses)
  audio_in.py    capture micro 16 kHz mono (sounddevice) + VAD Silero
  stt.py         faster-whisper (large-v3-turbo, float16, CUDA, language="fr"), modèle chargé UNE fois au démarrage
  brain.py       client Anthropic en streaming, personnalité, historique
  chunker.py     découpe le flux de tokens en phrases et extrait les balises d'émotion
  tts.py         client WebSocket Fish Audio (connexion ouverte en permanence) + secours HTTP en streaming
  audio_out.py   lecture PCM en flux (sounddevice OutputStream), arrêt instantané, mesure du volume (RMS)
  fillers.py     petits sons de réflexion (« hmm », « alors… ») pré-générés et mis en cache
  events.py      serveur WebSocket local (ws://127.0.0.1:8765) qui diffuse des événements JSON
  latency.py     chronométrage de chaque étape, une ligne de journal par tour de parole
persona.md       la personnalité (texte libre, modifiable sans toucher au code)
config.toml      réglages : modèles, voix, seuils VAD, format des balises, taille d'historique…
tests/           tests pytest
```

## Détails par composant

### 1. Écoute : VAD et STT
- Silero VAD (paquet `silero-vad`) sur des trames de 30 ms environ. La fin de parole est détectée après **~400 ms de silence** (réglable). C'est le réglage le plus important pour la latence.
- Garde un pré-tampon d'environ 300 ms pour ne pas couper le début de mes phrases.
- faster-whisper : `large-v3-turbo`, `compute_type="float16"`, `beam_size=1`, `language="fr"`, `vad_filter=False` (le VAD est déjà fait). Fais un « préchauffage » au démarrage (transcription d'1 s de silence) pour éviter la lenteur du premier appel.
- Sous Windows, faster-whisper/CTranslate2 a besoin de cuBLAS et cuDNN pour CUDA 12. Documente l'installation dans le README (paquets pip `nvidia-cublas-cu12` et `nvidia-cudnn-cu12`, ou les DLL dans le PATH) et vérifie au démarrage que le GPU est bien utilisé, avec un message clair sinon.

### 2. Cerveau : Claude
- SDK Python `anthropic`, appel en **streaming** (`client.messages.stream`).
- Modèle réglable dans `config.toml`. Par défaut `claude-haiku-5-5`, le plus rapide ; `claude-sonnet-5-5` en option, pour une personnalité plus riche.
- **Prompt caching** : la consigne système (personnalité + règles) est marquée `cache_control: {"type": "ephemeral"}` pour réduire le délai avant le premier token.
- La consigne système, construite à partir de `persona.md`, impose :
  - des réponses **orales** : courtes (1 à 3 phrases le plus souvent), sans markdown, sans listes, sans emojis, sans astérisques ;
  - la première phrase **courte**, pour que l'audio démarre vite ;
  - **une balise d'émotion au début de chaque phrase** quand c'est pertinent, uniquement parmi la liste autorisée fournie dans la consigne ;
  - le français parlé, avec la personnalité définie.
- Historique : les N derniers échanges (réglable, 20 par défaut).

### 3. Émotions
- Le format des balises dépend du modèle Fish et se règle dans la config : `(happy)` pour le modèle S1, `[happy]` pour S2-Pro. **Vérifie le format exact et la liste des balises autorisées dans la doc officielle actuelle** (https://docs.fish.audio, page « Emotion reference »).
- Mets la liste blanche des balises dans `config.toml` (par exemple : happy, sad, angry, excited, surprised, calm, sarcastic, worried, laughing, whispering, shouting, soft tone…).
- Le chunker :
  - laisse les balises valides dans le texte envoyé à Fish, qui s'en sert pour l'intonation ;
  - **supprime** les balises inconnues et tout résidu de markdown ;
  - émet un événement `emotion` (pour l'avatar) quand une balise est rencontrée.
- Pas d'appel IA supplémentaire pour les émotions : elles viennent du même flux.

### 4. Chunker (le cœur de la réactivité)
- Coupe sur `. ! ? …` et les retours à la ligne, sans couper les nombres décimaux ni les abréviations courantes (« M. », « etc. »).
- **Première phrase : mode agressif.** Si aucune fin de phrase n'arrive avant environ 60 caractères, coupe sur la prochaine virgule. Les phrases suivantes peuvent être regroupées jusqu'à environ 200 caractères pour une meilleure prosodie.
- En fin de flux, vide ce qui reste.
- Code pur, sans entrées/sorties, donc entièrement testable.

### 5. Voix : Fish Audio
- **WebSocket** `wss://api.fish.audio/v1/tts/live` : sérialisation MessagePack, événement `start` avec la config (reference_id, format `pcm`, `latency: "balanced"`, fréquence d'échantillonnage), puis événements `text` phrase par phrase, `flush` après chaque phrase pour forcer la génération, et `stop` en fin de réponse. Le modèle est passé dans un en-tête. **Vérifie le protocole exact dans la doc actuelle** (page « WebSocket TTS live » et le SDK Python officiel). Utilise le SDK officiel s'il expose bien le streaming WebSocket avec flush, sinon implémente le protocole avec `websockets` + `msgpack`.
- Ouvre la connexion au démarrage et garde-la active : reconnexion automatique avec un délai croissant si elle tombe.
- **Secours** : si le WebSocket échoue, bascule sur l'endpoint HTTP en streaming, phrase par phrase, et journalise l'incident.
- Sortie en **PCM brut** (pas de MP3 : pas de décodage, donc moins de délai). La fréquence d'échantillonnage doit être la même côté Fish et côté lecture.

### 6. Lecture audio
- `sounddevice.OutputStream` alimenté par une file. Lecture dès les premiers octets.
- `stop()` vide la file et coupe le son en moins de 50 ms.
- Calcule le volume RMS environ 30 fois par seconde et l'envoie sur le bus (`speaking_level`), pour la synchronisation des lèvres du futur avatar.
- Garde la trace de **quelles phrases ont réellement été jouées** (entièrement ou en partie).

### 7. Interruption (barge-in)
- Le micro et le VAD **restent actifs pendant qu'elle parle**.
- Si je parle pendant sa réponse, avec un seuil plus strict (probabilité VAD plus haute et au moins ~300 ms de parole continue, réglables) :
  1. arrêt immédiat de la lecture ;
  2. annulation du flux Claude en cours (annulation de la tâche asyncio) ;
  3. arrêt de la génération Fish en cours ;
  4. **l'historique ne garde que ce qui a réellement été prononcé**, suivi d'un marqueur du genre « [interrompu] » ;
  5. ma nouvelle phrase est transcrite et traitée normalement.
- Par défaut on suppose un casque. Avec des haut-parleurs, documente le risque d'auto-interruption (le micro capte sa voix) et prévois une option `speakers_mode` qui durcit encore les seuils. L'annulation d'écho est hors périmètre pour la V1.

### 8. Sons de réflexion (masquage de latence)
- Au premier lancement, génère via Fish 5 à 8 courtes interjections avec la voix choisie (« Hmm… », « Alors… », « Attends… », « Ah ! ») et mets-les en cache dans `cache/fillers/` en PCM.
- Si aucun audio de réponse n'est arrivé environ 700 ms (réglable) après la fin de ma phrase, joue un son au hasard, sans répéter le dernier utilisé.
- Désactivable dans la config.

### 9. Bus d'événements (préparation de l'avatar, phase 2)
- Serveur WebSocket local qui diffuse des événements JSON : `listening`, `user_speech_start`, `user_speech_end`, `transcript`, `thinking`, `emotion`, `speaking_start`, `speaking_level`, `speaking_end`, `interrupted`.
- L'avatar lui-même (VTube Studio/Live2D ou VRM web) est **hors périmètre** de la V1. Fournis seulement un petit script `tools/event_monitor.py` qui affiche les événements en direct.

### 10. Mesure de la latence
À chaque tour, affiche une ligne comme :
```
[tour 12] fin_parole→STT 290ms | STT→1er_token 480ms | 1er_token→1ère_phrase 120ms | 1ère_phrase→1er_audio 350ms | TOTAL 1240ms
```
Écris aussi ces mesures dans `logs/latency.csv` pour pouvoir comparer les réglages.

## Robustesse
- Aucune exception ne doit tuer la boucle de conversation. Journalise l'erreur, dis une courte phrase de secours (un son pré-généré, par exemple) et repars en écoute.
- Claude en erreur ou trop lent (plus de 5 s sans premier token) : phrase de secours et reprise.
- Ctrl+C : arrêt propre (fermeture des flux audio et des WebSockets).
- Journalisation avec `logging` ; niveau DEBUG activable dans la config.

## Tests (pytest)
- `chunker` : découpe des phrases, mode agressif sur la première phrase, abréviations et décimaux, balises valides gardées, balises invalides supprimées, markdown nettoyé, vidage final.
- Historique : troncature correcte après une interruption (seul le texte prononcé est gardé).
- Test d'intégration de l'orchestrateur avec de **fausses** implémentations de STT, LLM, TTS et audio (interfaces/protocoles injectables). Il vérifie que la première phrase part vers le TTS avant la fin du flux LLM, et qu'une interruption annule bien tout.
- Chaque composant externe (Claude, Fish, micro, haut-parleur) est derrière une interface, pour pouvoir le remplacer par une version factice dans les tests.

## Ordre de réalisation (chaque étape doit fonctionner avant de passer à la suivante)
1. **Squelette** : projet, `config.toml`, `.env.example`, `.gitignore`, `requirements.txt` (ou `pyproject.toml`), chunker + ses tests.
2. **Voix seule** : `tools/say.py "texte"`, qui parle via Fish en WebSocket avec lecture en flux, et mesure du délai jusqu'au premier audio.
3. **Conversation au clavier** : je tape, Claude répond en streaming, elle parle phrase par phrase. Mesure de latence.
4. **Micro** : VAD + Whisper GPU, conversation complète en mains libres.
5. **Interruption** (barge-in) avec troncature de l'historique.
6. **Sons de réflexion** + bus d'événements + `event_monitor.py`.
7. **Réglage fin** : à partir de `latency.csv`, propose-moi des ajustements (silence VAD, taille de la première phrase, modèle Claude…).

À la fin de chaque étape, montre-moi comment la tester, puis attends ma validation avant de continuer.

## Règles de travail
- Si la doc actuelle de Fish Audio ou d'Anthropic contredit ce prompt (nom d'endpoint, format des événements, ID de modèle), **c'est la doc qui a raison** : adapte-toi et signale-moi l'écart.
- Ne remplace pas un composant par une autre solution (autre TTS, autre LLM, API voix-à-voix) sans me demander.
- Pas de fonctionnalités en plus de celles décrites (pas de mémoire longue durée, pas d'interface graphique, pas d'avatar) : ce sera pour plus tard.
- README en français : installation (dont CUDA/cuDNN sous Windows), configuration, lancement, réglages de latence, dépannage micro et GPU.
