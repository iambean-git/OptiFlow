import { useState } from 'react'
export default function FreeTest() {


  return (
    <div class="relative overflow-x-auto shadow-md sm:rounded-lg">
      <table class="w-full text-sm text-left rtl:text-right text-gray-500">
        <thead class="text-xs text-gray-700 uppercase bg-gray-50 dark:bg-gray-700 ">
          <tr>
            <th scope="col" class="px-6 py-3">
              번호
            </th>
            <th scope="col" class="px-6 py-3">
              이름
            </th>
            <th scope="col" class="px-6 py-3">
              지역
            </th>
            <th scope="col" class="px-6 py-3">
              메일
            </th>
            <th scope="col" class="px-6 py-3">
              일시
            </th>
            <th scope="col" class="px-6 py-3">
              문의사항
            </th>
            <th scope="col" class="px-6 py-3">
            </th>
          </tr>
        </thead>
        <tbody>
          <tr class="bg-white border-b  border-gray-200 hover:bg-gray-50  font-semibold">
            <th scope="row" class="px-6 py-4 font-medium text-gray-900 whitespace-nowrap ">
              1
            </th>
            <td class="px-6 py-4">
              조은빈
            </td>
            <td class="px-6 py-4">
              부산
            </td>
            <td class="px-6 py-4">
              abcd1234@gmail.com
            </td>
            <td class="px-6 py-4">
              2025-02-12 09:47
            </td>
            <td class="px-6 py-4">
              없습니다.
            </td>
            <td class="px-6 py-4">
              <span className="border rounded-md px-2 py-1">new</span>
            </td>
          </tr>
          <tr class="bg-white border-b  border-gray-200 hover:bg-gray-50 ">
            <th scope="row" class="px-6 py-4 font-medium text-gray-900 whitespace-nowrap ">
              1
            </th>
            <td class="px-6 py-4">
              조은빈
            </td>
            <td class="px-6 py-4">
              부산
            </td>
            <td class="px-6 py-4">
              abcd1234@gmail.com
            </td>
            <td class="px-6 py-4">
              2025-02-12 09:47
            </td>
            <td class="px-6 py-4">
              없습니다.
            </td>
            <td class="px-6 py-4">
            </td>
          </tr>
          <tr class="bg-white border-b  border-gray-200 hover:bg-gray-50 ">
            <th scope="row" class="px-6 py-4 font-medium text-gray-900 whitespace-nowrap ">
              1
            </th>
            <td class="px-6 py-4">
              조은빈
            </td>
            <td class="px-6 py-4">
              부산
            </td>
            <td class="px-6 py-4">
              abcd1234@gmail.com
            </td>
            <td class="px-6 py-4">
              2025-02-12 09:47
            </td>
            <td class="px-6 py-4">
              없습니다.
            </td>
            <td class="px-6 py-4">
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  )
}
