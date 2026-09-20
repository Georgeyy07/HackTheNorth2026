# Roadscope Mobile (React Native + Expo)

A React Native mobile application built with **Expo** to monitor and manage road hazards in real time, backed by **Tiger Data (TimescaleDB / PostgreSQL)**.

---

## Features

- **Interactive Pothole Map**: Displays reported road potholes with color-coded severity markers (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`).
- **Drag-to-Move**: Long-press and drag any marker pin across the map to automatically update its GPS coordinates in the database.
- **Tap-to-Report**: Tap any location on the map (or click "+ Report Hazard") to register a new pothole.
- **Full CRUD Support**:
  - **Create**: Report hazard with coordinates and severity level.
  - **Read**: Live list of hazards with filtering chips (`ALL`, `CRITICAL`, `HIGH`, etc.).
  - **Update**: Edit severity and move coordinates.
  - **Delete**: Remove pothole records with a single tap.
- **Database Seeding**: Easily seed default sample hazards with the "Seed DB" button.
- **Configurable Backend Host**: Built-in settings modal to change the backend API URL on the fly (useful when testing on a physical phone over Wi-Fi).

---

## Getting Started

### 1. Ensure the Python Backend is Running

From the repository root:

```powershell
python -m road_viewer.server --host 0.0.0.0
```

*(Note: `--host 0.0.0.0` allows mobile devices on your Wi-Fi or emulators to connect).*

---

### 2. Install Dependencies

Open a terminal in the `frontend` folder:

```powershell
cd frontend
npm install
```

---

### 3. Start Expo

```powershell
npx expo start
```

- **Physical Device (iPhone / Android)**: Download the **Expo Go** app from the App Store or Google Play. Scan the QR code displayed in the terminal or browser.
  - *Tip*: In the app, tap the `⚙️` settings icon and set the Server URL to your computer's local Wi-Fi IP address (e.g. `http://192.168.1.50:8765`).
- **Android Emulator**: Press `a` in the terminal. (Default URL `http://10.0.2.2:8765` connects automatically).
- **iOS Simulator**: Press `i` in the terminal. (Default URL `http://localhost:8765` connects automatically).
- **Web Browser**: Press `w` in the terminal or run:
  ```powershell
  npx expo start --web
  ```

---

## Project Structure

```
frontend/
├── package.json               # Expo & React Native dependencies
├── app.json                   # Expo project configuration
├── index.js                   # Root component registration
├── tsconfig.json              # TypeScript configuration
├── App.tsx                    # Main app view, navigation, and state
└── src/
    ├── api/
    │   └── client.ts          # Tiger Data REST API client
    ├── components/
    │   ├── PotholeMap.tsx     # Map with custom draggable pins & web fallback
    │   ├── PotholeList.tsx    # Collapsible hazard list drawer
    │   ├── AddPotholeModal.tsx # Report hazard dialog
    │   └── EditPotholeModal.tsx # Edit hazard dialog
    ├── constants/
    │   └── theme.ts           # Dark mode styling & severity colors
    └── types/
        └── pothole.ts         # Pothole & Severity TypeScript types
```
