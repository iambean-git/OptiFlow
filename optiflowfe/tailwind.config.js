/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      width: {
        '128': '32rem',
      },
      fontFamily: {
        PeoplefirstILTTF : 'PeoplefirstILTTF',
        Freesentation : 'Freesentation',
        Pretendard : 'Pretendard',
        SUIT : 'SUIT'
      }
    },
  },
  plugins: [],
}
