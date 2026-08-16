import axios from 'axios';

// The UNO Q recognition-only edition has no application backend.
// Keep this export only for legacy screens that are not part of the flow.
export const api = axios.create();
