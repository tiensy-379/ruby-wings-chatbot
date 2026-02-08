const { google } = require('googleapis');

async function test() {
  const auth = new google.auth.GoogleAuth({
    credentials: JSON.parse(process.env.GOOGLE_SERVICE_ACCOUNT_JSON),
    scopes: ['https://www.googleapis.com/auth/spreadsheets'],
  });

  const sheets = google.sheets({ version: 'v4', auth });

  const res = await sheets.spreadsheets.get({
    spreadsheetId: '1SdVbwkuxb8l1meEW--ddyfh4WmUvSXXMOPQ5bCyPkdk',
  });

  console.log('OK:', res.data.properties.title);
}

test().catch(err => console.error('ERR:', err.message));
