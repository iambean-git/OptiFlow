import { FaCaretDown } from "react-icons/fa6";
import React, { useState, useRef } from 'react';
import '../css/customSelectBox.css';

export default function CustomSelectBox({ options, selectLabel, selectedOption, setSelectedOption, disabled, size }) {
    const [isOpen, setIsOpen] = useState(false); // 드롭다운 열림 상태
    // const [selectedOption, setSelectedOption] = useState(null); // 선택된 옵션
    const selectRef = useRef(null);

    // 드롭다운 열기/닫기 토글
    const toggleDropdown = () => {
        setIsOpen(!isOpen);
    };

    // 옵션 선택 처리
    const handleSelect = (option) => {
        setSelectedOption(option);
        setIsOpen(false); // 드롭다운 닫기
    };

    // 드롭다운 외부 클릭 시 닫기
    const handleClickOutside = (event) => {
        if (selectRef.current && !selectRef.current.contains(event.target)) {
            setIsOpen(false);
        }
    };

    // 외부 클릭 감지 이벤트 등록
    React.useEffect(() => {
        document.addEventListener('click', handleClickOutside);
        return () => {
            document.removeEventListener('click', handleClickOutside);
        };
    }, []);

    return (
        <div className={`custom-select ${size ? size : "w-[250px]"}`} ref={selectRef}>
            <button className={`flex h-full justify-between shadow-inner select-trigger`}
                onClick={toggleDropdown} disabled={disabled ? disabled : false}>
                <span className="">
                    {selectedOption ? selectedOption.label : selectLabel}
                </span>
                <span className="h-full flex justify-center items-center"><FaCaretDown /></span>
            </button>
            {isOpen && (
                <ul className="select-options ">
                    {options.map((option) => (
                        <li
                            key={option.value}
                            className="select-option"
                            onClick={() => handleSelect(option)}
                        >
                            {option.label}
                        </li>
                    ))}
                </ul>
            )}
        </div>
    );
}