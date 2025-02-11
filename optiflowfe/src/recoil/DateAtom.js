import { atom, selector } from "recoil";


export const NowDate = atom({
    key: "NowDate",
    default: (() => {
        const now = new Date(); // 현재 시간 가져오기
        const customDate = new Date(2023, 9, 21); // 원하는 날짜 설정 (월은 0부터 시작)

        // 현재 시간으로 맞추기
        customDate.setHours(now.getHours(), now.getMinutes(), now.getSeconds(), now.getMilliseconds());

        return customDate;
    })(),
});

export const maxDate10am = selector({
    key: "maxDate10am",
    get: ({ get }) => {
        const now = get(MaxDate); // 현재 날짜 가져오기
        const maxDate10am = new Date(now); // 새로운 Date 객체 생성 (원본 변경 방지)

        maxDate10am.setHours(10, 0, 0, 0); // 오전 10시로 설정
        return maxDate10am;
    }
});

export const MaxDate = atom({
    key: "MaxDate",
    default: new Date(2024, 9, 16, 23, 59, 59)
})

// export const AtomN2 = selector({
//     key : "AtomN2",
//    get : ({get}) => {
//     return get(AtomN) * 2;
//    }
// });