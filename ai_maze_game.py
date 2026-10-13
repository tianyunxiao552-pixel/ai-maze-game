"""
AI 迷宫追逐小游戏
-------------------------------------------------
运行方式：python ai_maze_game.py
依赖：Python 标准库 tkinter，无需额外安装第三方库。

玩法说明：
1. 方向键或 WASD 控制蓝色玩家移动。
2. 收集黄色能量点获得分数。
3. 红色/紫色 AI 敌人会使用搜索算法追踪玩家。
4. 绿色出口会在收集足够能量后打开，到达出口即可进入下一关。
5. 按 P 暂停，按 R 重新开始。

人工智能设计：
- 敌人不是简单直线移动，而是把迷宫看成状态空间图。
- 使用 BFS 搜索从敌人位置到目标位置的最短路径。
- 不同敌人采用不同目标策略：追踪当前位置、预测未来位置、守门巡逻。
- 随着关卡提升，敌人移动速度和决策频率会增加。
"""

from __future__ import annotations

import random
import time
import tkinter as tk
from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

Position = Tuple[int, int]


@dataclass
class Config:
    """游戏全局参数。"""

    cell_size: int = 32
    rows: int = 17
    cols: int = 21
    margin_top: int = 70
    margin_left: int = 20
    fps_ms: int = 80
    player_lives: int = 3
    win_collect_count: int = 12

    @property
    def width(self) -> int:
        return self.margin_left * 2 + self.cols * self.cell_size

    @property
    def height(self) -> int:
        return self.margin_top + 20 + self.rows * self.cell_size


class Maze:
    """负责迷宫地图、墙体、能量点和出口。"""

    RAW_MAP = [
        "#####################",
        "#P....#......#......#",
        "#.###.#.####.#.####.#",
        "#...#.#....#.#....#.#",
        "###.#.####.#.####.#.#",
        "#...#......#......#.#",
        "#.########.######.#.#",
        "#.........E......#..#",
        "#.#####.#####.#####.#",
        "#.....#.....#.......#",
        "#####.###.#.###.#####",
        "#.....#...#...#.....#",
        "#.#####.#####.###...#",
        "#.......#.....#.....#",
        "#.#######.###.###..X#",
        "#.................#.#",
        "#####################",
    ]

    def __init__(self, config: Config) -> None:
        self.config = config
        self.walls: Set[Position] = set()
        self.floor: Set[Position] = set()
        self.items: Set[Position] = set()
        self.player_start: Position = (1, 1)
        self.enemy_starts: List[Position] = []
        self.exit_pos: Position = (15, 19)
        self.parse_map()

    def parse_map(self) -> None:
        self.walls.clear()
        self.floor.clear()
        self.items.clear()
        self.enemy_starts.clear()
        for r, line in enumerate(self.RAW_MAP):
            for c, char in enumerate(line):
                pos = (r, c)
                if char == "#":
                    self.walls.add(pos)
                    continue
                self.floor.add(pos)
                if char == "P":
                    self.player_start = pos
                elif char == "E":
                    self.enemy_starts.append(pos)
                elif char == "X":
                    self.exit_pos = pos
                else:
                    if char == ".":
                        self.items.add(pos)

    def reset_items(self, level: int) -> None:
        """根据关卡重置能量点，关卡越高能量点越密。"""
        self.items.clear()
        random.seed(100 + level)
        candidates = [p for p in self.floor if p not in {self.player_start, self.exit_pos}]
        random.shuffle(candidates)
        item_count = min(len(candidates), self.config.win_collect_count + level * 3)
        for pos in candidates[:item_count]:
            if pos not in self.enemy_starts:
                self.items.add(pos)

    def is_wall(self, pos: Position) -> bool:
        return pos in self.walls

    def in_bounds(self, pos: Position) -> bool:
        r, c = pos
        return 0 <= r < self.config.rows and 0 <= c < self.config.cols

    def is_walkable(self, pos: Position) -> bool:
        return self.in_bounds(pos) and pos not in self.walls

    def neighbors(self, pos: Position) -> Iterable[Position]:
        r, c = pos
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = (r + dr, c + dc)
            if self.is_walkable(nxt):
                yield nxt

    def to_pixel(self, pos: Position) -> Tuple[int, int, int, int]:
        r, c = pos
        s = self.config.cell_size
        x1 = self.config.margin_left + c * s
        y1 = self.config.margin_top + r * s
        return x1, y1, x1 + s, y1 + s

    def cell_center(self, pos: Position) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.to_pixel(pos)
        return (x1 + x2) // 2, (y1 + y2) // 2


class SearchAI:
    """敌人使用的图搜索算法。"""

    @staticmethod
    def bfs_next_step(maze: Maze, start: Position, goal: Position) -> Optional[Position]:
        """BFS 求从 start 到 goal 的下一步。找不到路径时返回 None。"""
        if start == goal:
            return start
        queue: deque[Position] = deque([start])
        parent: Dict[Position, Optional[Position]] = {start: None}
        while queue:
            current = queue.popleft()
            if current == goal:
                break
            for nxt in maze.neighbors(current):
                if nxt not in parent:
                    parent[nxt] = current
                    queue.append(nxt)
        if goal not in parent:
            return None
        step = goal
        while parent[step] is not None and parent[step] != start:
            step = parent[step]  # type: ignore[index]
        return step

    @staticmethod
    def shortest_distance(maze: Maze, start: Position, goal: Position) -> int:
        """返回两个格子之间的最短步数，用于敌人决策评价。"""
        if start == goal:
            return 0
        queue: deque[Tuple[Position, int]] = deque([(start, 0)])
        seen = {start}
        while queue:
            current, dist = queue.popleft()
            for nxt in maze.neighbors(current):
                if nxt in seen:
                    continue
                if nxt == goal:
                    return dist + 1
                seen.add(nxt)
                queue.append((nxt, dist + 1))
        return 9999

    @staticmethod
    def safest_neighbor(maze: Maze, start: Position, enemies: List[Position]) -> Position:
        """给玩家提示使用：找一个离所有敌人都更远的相邻格。"""
        best = start
        best_score = -1
        for candidate in [start, *list(maze.neighbors(start))]:
            score = min(SearchAI.shortest_distance(maze, candidate, e) for e in enemies)
            if score > best_score:
                best = candidate
                best_score = score
        return best


@dataclass
class Player:
    pos: Position
    lives: int
    score: int = 0
    collected: int = 0
    direction: Position = (0, 1)

    def reset_position(self, start: Position) -> None:
        self.pos = start
        self.direction = (0, 1)


class Enemy:
    """敌人智能体。"""

    def __init__(self, name: str, pos: Position, color: str, mode: str) -> None:
        self.name = name
        self.pos = pos
        self.start = pos
        self.color = color
        self.mode = mode
        self.sleep_counter = 0
        self.patrol_index = 0
        self.patrol_points: List[Position] = [(7, 9), (1, 18), (15, 1), (15, 18)]

    def reset(self) -> None:
        self.pos = self.start
        self.sleep_counter = 0
        self.patrol_index = 0

    def choose_target(self, maze: Maze, player: Player, level: int) -> Position:
        """根据不同 AI 策略选择目标。"""
        if self.mode == "chaser":
            return player.pos
        if self.mode == "ambusher":
            pr, pc = player.pos
            dr, dc = player.direction
            target = (pr + dr * (2 + level // 2), pc + dc * (2 + level // 2))
            while not maze.is_walkable(target):
                target = (target[0] - dr, target[1] - dc)
                if target == player.pos:
                    break
            return target
        if self.mode == "guardian":
            if SearchAI.shortest_distance(maze, self.pos, player.pos) <= 6 + level:
                return player.pos
            return (13, 19)
        point = self.patrol_points[self.patrol_index % len(self.patrol_points)]
        if self.pos == point:
            self.patrol_index += 1
        return point

    def update(self, maze: Maze, player: Player, level: int) -> None:
        """按照搜索结果移动一步。"""
        delay = max(1, 5 - level)
        self.sleep_counter += 1
        if self.sleep_counter < delay:
            return
        self.sleep_counter = 0
        target = self.choose_target(maze, player, level)
        step = SearchAI.bfs_next_step(maze, self.pos, target)
        if step is not None and step != self.pos:
            self.pos = step
        else:
            choices = list(maze.neighbors(self.pos))
            if choices:
                self.pos = random.choice(choices)


class GameStats:
    """记录每局游戏数据，方便总结运行效果。"""

    def __init__(self) -> None:
        self.start_time = time.time()
        self.moves = 0
        self.ai_decisions = 0
        self.levels_finished = 0

    def reset(self) -> None:
        self.start_time = time.time()
        self.moves = 0
        self.ai_decisions = 0
        self.levels_finished = 0

    def elapsed(self) -> int:
        return int(time.time() - self.start_time)


class GameApp:
    """Tkinter 图形界面与主循环。"""

    def __init__(self) -> None:
        self.config = Config()
        self.maze = Maze(self.config)
        self.root = tk.Tk()
        self.root.title("AI 迷宫追逐战")
        self.root.resizable(False, False)
        self.canvas = tk.Canvas(
            self.root,
            width=self.config.width,
            height=self.config.height,
            bg="#111827",
            highlightthickness=0,
        )
        self.canvas.pack()
        self.player = Player(self.maze.player_start, self.config.player_lives)
        self.enemies: List[Enemy] = []
        self.level = 1
        self.running = True
        self.game_over = False
        self.message = "方向键/WASD移动，收集能量点，躲避AI敌人"
        self.stats = GameStats()
        self.create_enemies()
        self.maze.reset_items(self.level)
        self.bind_keys()
        self.draw()
        self.root.after(self.config.fps_ms, self.tick)

    def create_enemies(self) -> None:
        starts = self.maze.enemy_starts or [(7, 9)]
        self.enemies = [
            Enemy("BFS追踪者", starts[0], "#ef4444", "chaser"),
            Enemy("预测拦截者", (9, 11), "#a855f7", "ambusher"),
            Enemy("出口守卫", (13, 15), "#f97316", "guardian"),
        ]

    def bind_keys(self) -> None:
        self.root.bind("<Up>", lambda _: self.try_move((-1, 0)))
        self.root.bind("<Down>", lambda _: self.try_move((1, 0)))
        self.root.bind("<Left>", lambda _: self.try_move((0, -1)))
        self.root.bind("<Right>", lambda _: self.try_move((0, 1)))
        self.root.bind("w", lambda _: self.try_move((-1, 0)))
        self.root.bind("s", lambda _: self.try_move((1, 0)))
        self.root.bind("a", lambda _: self.try_move((0, -1)))
        self.root.bind("d", lambda _: self.try_move((0, 1)))
        self.root.bind("p", lambda _: self.toggle_pause())
        self.root.bind("P", lambda _: self.toggle_pause())
        self.root.bind("r", lambda _: self.restart())
        self.root.bind("R", lambda _: self.restart())

    def try_move(self, delta: Position) -> None:
        if self.game_over or not self.running:
            return
        r, c = self.player.pos
        dr, dc = delta
        nxt = (r + dr, c + dc)
        self.player.direction = delta
        if self.maze.is_walkable(nxt):
            self.player.pos = nxt
            self.stats.moves += 1
            self.collect_item()
            self.check_exit()
            self.check_collision()
            self.draw()

    def collect_item(self) -> None:
        if self.player.pos in self.maze.items:
            self.maze.items.remove(self.player.pos)
            self.player.score += 10
            self.player.collected += 1
            self.message = f"获得能量点！当前已收集 {self.player.collected} 个"

    def check_exit(self) -> None:
        if self.player.pos != self.maze.exit_pos:
            return
        if self.player.collected >= self.config.win_collect_count:
            self.level_up()
        else:
            left = self.config.win_collect_count - self.player.collected
            self.message = f"出口还未开启，还需要 {left} 个能量点"

    def level_up(self) -> None:
        self.level += 1
        self.stats.levels_finished += 1
        self.player.score += 100
        self.player.collected = 0
        self.player.reset_position(self.maze.player_start)
        self.maze.reset_items(self.level)
        for enemy in self.enemies:
            enemy.reset()
        self.message = f"进入第 {self.level} 关：AI决策速度提升"

    def check_collision(self) -> None:
        enemy_positions = [enemy.pos for enemy in self.enemies]
        if self.player.pos not in enemy_positions:
            return
        self.player.lives -= 1
        if self.player.lives <= 0:
            self.game_over = True
            self.running = False
            self.message = "游戏结束：AI敌人抓住了你。按R重新开始。"
        else:
            self.message = f"被AI敌人抓住，剩余生命 {self.player.lives}"
            self.player.reset_position(self.maze.player_start)
            for enemy in self.enemies:
                enemy.reset()

    def toggle_pause(self) -> None:
        if self.game_over:
            return
        self.running = not self.running
        self.message = "已暂停，按P继续" if not self.running else "继续游戏"
        self.draw()

    def restart(self) -> None:
        self.level = 1
        self.player = Player(self.maze.player_start, self.config.player_lives)
        self.maze.reset_items(self.level)
        for enemy in self.enemies:
            enemy.reset()
        self.running = True
        self.game_over = False
        self.message = "新一局开始：方向键/WASD移动"
        self.stats.reset()
        self.draw()

    def tick(self) -> None:
        if self.running and not self.game_over:
            for enemy in self.enemies:
                enemy.update(self.maze, self.player, self.level)
                self.stats.ai_decisions += 1
            self.check_collision()
            self.draw()
        self.root.after(self.config.fps_ms, self.tick)

    def draw(self) -> None:
        self.canvas.delete("all")
        self.draw_header()
        self.draw_maze()
        self.draw_items()
        self.draw_exit()
        self.draw_hint()
        self.draw_player()
        self.draw_enemies()
        if self.game_over:
            self.draw_center_message("GAME OVER", "按 R 重新开始")
        elif not self.running:
            self.draw_center_message("PAUSED", "按 P 继续")

    def draw_header(self) -> None:
        self.canvas.create_rectangle(0, 0, self.config.width, self.config.margin_top, fill="#0f172a", outline="")
        title = "AI 迷宫追逐战：BFS路径搜索 + 智能体决策"
        self.canvas.create_text(20, 18, text=title, anchor="w", fill="#f8fafc", font=("Arial", 16, "bold"))
        info = (
            f"关卡:{self.level}  分数:{self.player.score}  生命:{self.player.lives}  "
            f"收集:{self.player.collected}/{self.config.win_collect_count}  "
            f"用时:{self.stats.elapsed()}s  移动:{self.stats.moves}"
        )
        self.canvas.create_text(20, 47, text=info, anchor="w", fill="#cbd5e1", font=("Arial", 11))
        self.canvas.create_text(
            self.config.width - 20,
            47,
            text="P暂停 / R重开",
            anchor="e",
            fill="#93c5fd",
            font=("Arial", 11),
        )

    def draw_maze(self) -> None:
        for r in range(self.config.rows):
            for c in range(self.config.cols):
                pos = (r, c)
                x1, y1, x2, y2 = self.maze.to_pixel(pos)
                if self.maze.is_wall(pos):
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="#334155", outline="#1e293b")
                else:
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="#0b1220", outline="#111827")

    def draw_items(self) -> None:
        for pos in self.maze.items:
            x, y = self.maze.cell_center(pos)
            self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill="#fde047", outline="")

    def draw_exit(self) -> None:
        x1, y1, x2, y2 = self.maze.to_pixel(self.maze.exit_pos)
        open_exit = self.player.collected >= self.config.win_collect_count
        color = "#22c55e" if open_exit else "#166534"
        self.canvas.create_rectangle(x1 + 5, y1 + 5, x2 - 5, y2 - 5, fill=color, outline="#bbf7d0", width=2)
        self.canvas.create_text((x1 + x2) // 2, (y1 + y2) // 2, text="出", fill="#f0fdf4", font=("Arial", 12, "bold"))

    def draw_hint(self) -> None:
        enemy_positions = [e.pos for e in self.enemies]
        safe_pos = SearchAI.safest_neighbor(self.maze, self.player.pos, enemy_positions)
        if safe_pos != self.player.pos:
            x1, y1, x2, y2 = self.maze.to_pixel(safe_pos)
            self.canvas.create_rectangle(x1 + 8, y1 + 8, x2 - 8, y2 - 8, outline="#38bdf8", width=2)

    def draw_player(self) -> None:
        x, y = self.maze.cell_center(self.player.pos)
        self.canvas.create_oval(x - 12, y - 12, x + 12, y + 12, fill="#3b82f6", outline="#bfdbfe", width=2)
        self.canvas.create_text(x, y, text="我", fill="white", font=("Arial", 10, "bold"))

    def draw_enemies(self) -> None:
        for enemy in self.enemies:
            x, y = self.maze.cell_center(enemy.pos)
            self.canvas.create_rectangle(x - 12, y - 12, x + 12, y + 12, fill=enemy.color, outline="#fee2e2", width=2)
            self.canvas.create_text(x, y, text="AI", fill="white", font=("Arial", 8, "bold"))
            self.canvas.create_text(x, y + 20, text=enemy.mode, fill="#e5e7eb", font=("Arial", 7))
        self.canvas.create_text(20, self.config.height - 10, text=self.message, anchor="w", fill="#f8fafc", font=("Arial", 10))

    def draw_center_message(self, main: str, sub: str) -> None:
        w = self.config.width
        h = self.config.height
        self.canvas.create_rectangle(w // 2 - 160, h // 2 - 70, w // 2 + 160, h // 2 + 70, fill="#020617", outline="#94a3b8", width=2)
        self.canvas.create_text(w // 2, h // 2 - 20, text=main, fill="#f8fafc", font=("Arial", 28, "bold"))
        self.canvas.create_text(w // 2, h // 2 + 25, text=sub, fill="#cbd5e1", font=("Arial", 14))

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    app = GameApp()
    app.run()


if __name__ == "__main__":
    main()
