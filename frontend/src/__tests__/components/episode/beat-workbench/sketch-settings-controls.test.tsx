// SPDX-License-Identifier: Elastic-2.0
// Copyright (c) 2026 ClaymoreLab
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { SketchAspectCheckbox } from "@/components/episode/beat-workbench/sketch-settings-controls";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string) => {
      if (key === "episode.sketchSettings.aspectRatio") return "画幅";
      return key;
    },
  }),
}));

describe("SketchAspectCheckbox", () => {
  it("offers all six aspect ratios with their crop cost", async () => {
    const user = userEvent.setup();
    const onAspectRatioChange = vi.fn();

    render(
      <SketchAspectCheckbox
        aspectRatio="16:9"
        onAspectRatioChange={onAspectRatioChange}
        flat
      />,
    );

    expect(screen.queryByRole("checkbox")).not.toBeInTheDocument();
    expect(screen.getByRole("combobox", { name: "画幅" })).toHaveTextContent("16:9");

    await user.click(screen.getByRole("combobox", { name: "画幅" }));
    // 六种画幅全部可选，且每项带裁切代价标注——所以按前缀匹配而非全等
    for (const ratio of ["9:16", "2:3", "3:4", "1:1", "4:3", "16:9"]) {
      expect(
        await screen.findByRole("option", {
          name: new RegExp(`^${ratio.replace(":", ":")}`),
        }),
      ).toBeInTheDocument();
    }
    // 与出片比例一致的那项要标成零裁切（测试环境不加载翻译文件，断言 key）
    expect(
      screen.getByRole("option", { name: /^9:16/ }),
    ).toHaveTextContent("aspectNoCrop");
    expect(
      screen.getByRole("option", { name: /^16:9/ }),
    ).toHaveTextContent("aspectCrop");

    await user.click(screen.getByRole("option", { name: /^2:3/ }));

    expect(onAspectRatioChange).toHaveBeenCalledWith("2:3");
  });
});
